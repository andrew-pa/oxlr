use std::{cell::RefCell, collections::HashMap, rc::Rc, borrow::Cow};
use itertools::Itertools;
use anyhow::*;

mod world;
mod value;
mod memory;

use world::World;
use value::*;
use memory::{Memory, Frame};

unsafe fn memcpy(src: *const u8, dest: *mut u8, size: usize) {
    let src_data = std::slice::from_raw_parts(src, size);
    let dest_data = std::slice::from_raw_parts_mut(dest, size);
    for (s, d) in src_data.iter().zip(dest_data.iter_mut()) {
        *d = *s;
    }
}

struct Machine<'w> {
    world: &'w World,
    mem: Memory<'w>,
}

impl<'w> Machine<'w> {
    fn new(world: &'w World) -> Machine {
        Machine {
            mem: Memory::new(world), world
        }
    }

    /// start the virtual machine
    fn start(&mut self, mut starting_module_path: ir::Path) {
        starting_module_path.0.push(ir::Symbol("start".into()));
        let (_, body) = self.world.get_function(&starting_module_path)
            .expect("a start function is present");
        log::trace!("starting execution");
        let rv = self.call_fn(body, vec![]).unwrap();
        println!("{} returned: {:?}", starting_module_path, rv);
    }

    /// look up and call a function by interpreting its body to determine the return value
    fn call_fn(&mut self, body: &'w ir::FnBody, args: Vec<Value>) -> Result<Value> {
        self.mem.stack.push(Frame::new(body.max_registers as usize));
        for (i, v) in args.into_iter().enumerate() {
            self.mem.cur_frame().store(&ir::code::Register(i as u32), v);
        }
        let mut cur_block_index = 0;
        let mut prev_block_index: Option<usize> = Some(0);
        'blocks: loop {
            let cur_block = &body.blocks[cur_block_index];
            for instr in cur_block.instrs.iter() {
                log::debug!("running instruction {:?}", instr);
                log::debug!("current frame {:?}", self.mem.cur_frame());
                use ir::code::Instruction;
                match instr {
                    Instruction::Phi(dest, precedents) => {
                        let res = self.mem.cur_frame().convert_value(&precedents[prev_block_index.as_ref().unwrap()]);
                        self.mem.cur_frame().store(dest, res)
                    },
                    Instruction::Br { cond, if_true, if_false } => {
                        // ostensibly this is the last instruction in the block
                        match self.mem.cur_frame().convert_value(cond) {
                            Value::Bool(true) => {
                                prev_block_index = Some(cur_block_index);
                                cur_block_index = *if_true;
                                continue 'blocks;
                            },
                            Value::Bool(false) => {
                                prev_block_index = Some(cur_block_index);
                                cur_block_index = *if_false;
                                continue 'blocks;
                            },
                            _ => bail!("expected bool")
                        }
                    },

                    Instruction::BinaryOp(op, dest, lhs, rhs) => {
                        use ir::code::BinOp;
                        let lhs = self.mem.cur_frame().convert_value(lhs);
                        let rhs = self.mem.cur_frame().convert_value(rhs);
                        let res = match (op, lhs, rhs) {
                            (BinOp::Add, Value::Int(a), Value::Int(b)) => Value::Int(a+b),
                            (BinOp::Sub, Value::Int(a), Value::Int(b)) => {
                                // do a saturating subtraction for now
                                // TODO: deal with overflow
                                if a.data < b.data {
                                    Value::Int(Integer::new(a.width, a.signed, 0))
                                } else {
                                    Value::Int(a-b)
                                }
                            },
                            (BinOp::Mul, Value::Int(a), Value::Int(b)) => Value::Int(a*b),
                            (BinOp::Div, Value::Int(a), Value::Int(b)) => Value::Int(a/b),
                            (BinOp::Eq,  a, b) => Value::Bool(a == b),
                            (BinOp::NEq,  a, b) => Value::Bool(a != b),
                            //TODO: implement the rest of the binary operators. for most of these,
                            //the operation also needs to be added to the corrosponding value as
                            //well (Integer/Float). Additionally, invalid/mismatched types should
                            //result in an actual error rather than panicking.
                            (op, lhs, rhs) => todo!("unimplemented binary operator {:?} ({:?}) {:?}", lhs, op, rhs)
                        };
                        self.mem.cur_frame().store(dest, res);
                    },
                    Instruction::UnaryOp(op, dest, inp) => {
                        use ir::code::UnaryOp;
                        let inp = self.mem.cur_frame().convert_value(inp);
                        let res = match (op, inp) {
                            (UnaryOp::LogNot, Value::Bool(v)) => Value::Bool(!v),
                            (UnaryOp::BitNot, Value::Int(v)) => Value::Int(v.bitwise_negate()),
                            (UnaryOp::Neg,    Value::Int(v)) if v.signed => Value::Int(v.negate()),
                            _ => bail!("invalid operand to unary operation")
                        };
                        self.mem.cur_frame().store(dest, res);
                    },

                    Instruction::LoadImm(dest, v) => {
                        let v = self.mem.cur_frame().convert_value(v);
                        self.mem.cur_frame().store(dest, v)
                    },
                    Instruction::LoadRef(dest, r#ref) => {
                        match self.mem.cur_frame().load(r#ref) {
                            Value::Ref(r) => self.mem.cur_frame().store(dest, r.value()),
                            v => bail!("expected ref, got: {:?}", v)
                        }
                    },
                    Instruction::StoreRef(dest, src) => {
                        match self.mem.cur_frame().load(dest) {
                            Value::Ref(r) => r.set_value(self.mem.cur_frame().convert_value(src)),
                            v => bail!("expected ref, got: {:?}", v)
                        }
                    },

                    Instruction::RefField(dest, src_ref, field) => {
                        match self.mem.cur_frame().load(src_ref) {
                            Value::Ref(r) => self.mem.cur_frame().store(dest,
                                Value::Ref(r.field(self.world, field)?)),
                            _ => bail!("expected ref")
                        }
                    }
                    Instruction::LoadField(dest, r#ref, field) => {
                        match self.mem.cur_frame().load(r#ref) {
                            Value::Ref(r) => self.mem.cur_frame().store(dest,
                                r.field(self.world, field)?.value()),
                            _ => bail!("expected ref")
                        }
                    },
                    Instruction::StoreField(src, r#ref, field) => {
                        match self.mem.cur_frame().load(r#ref) {
                            Value::Ref(r) => {
                                let val = self.mem.cur_frame().convert_value(src);
                                r.field(self.world, field)?.set_value(val)
                            },
                            _ => bail!("expected ref")
                        }
                    },

                    Instruction::RefIndex(dest, src_ref, index) => {
                        let index = match self.mem.cur_frame().convert_value(index) {
                            Value::Int(Integer { signed: false, data, .. }) => data as usize,
                            _ => bail!("invalid index")
                        };
                        match self.mem.cur_frame().load(src_ref) {
                            Value::Ref(r) =>
                                self.mem.cur_frame().store(dest,
                                    Value::Ref(r.indexed(self.world, index)?)),
                            _ => bail!("expected ref or array")
                        }
                    },
                    Instruction::LoadIndex(dest, r#ref, index) => {
                        let index = match self.mem.cur_frame().convert_value(index) {
                            Value::Int(Integer { signed: false, data, .. }) => data as usize,
                            _ => bail!("invalid index")
                        };
                        match self.mem.cur_frame().load(r#ref) {
                            Value::Ref(r) =>
                                self.mem.cur_frame().store(dest,
                                    r.indexed(self.world, index)?.value()),
                            _ => bail!("expected ref or array")
                        }
                    },
                    Instruction::StoreIndex(r#ref, index, src) => {
                        let index = match self.mem.cur_frame().convert_value(index) {
                            Value::Int(Integer { signed: false, data, .. }) => data as usize,
                            _ => bail!("invalid index")
                        };
                        match self.mem.cur_frame().load(r#ref) {
                            Value::Ref(r) => {
                                let val = self.mem.cur_frame().convert_value(src);
                                r.indexed(self.world, index)?.set_value(val);
                            },
                            _ => bail!("expected ref or array")
                        }
                    }

                    Instruction::Call(dest, fn_path, params) => {
                        log::trace!("calling {}", fn_path);
                        // TODO: Check types to make sure call is valid!
                        let (fn_sig, fn_body) = self.world.get_function(fn_path).ok_or_else(|| anyhow!("function not found"))?;
                        let params = params.iter().map(|p| self.mem.cur_frame().convert_value(p)).collect();
                        let result = self.call_fn(fn_body, params)?;
                        if let Some(dst) = dest {
                            self.mem.cur_frame().store(dst, result)
                        }
                    },
                    Instruction::CallImpl(dest, fn_path, params) => {
                        log::trace!("calling {}", fn_path);
                        // TODO: Check types to make sure call is valid!
                        let params: Vec<Value> = params.iter().map(|p| self.mem.cur_frame().convert_value(p)).collect();
                        let self_val = params.first().ok_or_else(|| anyhow!("call impl requires at least one parameter"))?;
                        let (fn_sig, fn_body) = self.world.find_impl(fn_path, &self_val.type_of(&self.mem))
                            .ok_or_else(|| anyhow!("implementation not found"))?;
                        let result = self.call_fn(fn_body, params)?;
                        if let Some(dst) = dest {
                            self.mem.cur_frame().store(dst, result)
                        }
                    },
                    Instruction::Return(v) => {
                        log::trace!("return");
                        let rv = self.mem.cur_frame().convert_value(v);
                        self.mem.pop_stack();
                        return Ok(rv)
                    },
                    Instruction::RefFunc(dest, _) => todo!(),
                    Instruction::UnwrapVariant(cond_dest, inner_ref_dest, src_val, variant_name) => {
                        match self.mem.cur_frame().load(src_val) {
                            Value::Ref(r) => {
                                let (ac_name, inner_ref) = r.unwrap_variant(self.world)?;
                                let valid_unwrap = ac_name == variant_name;
                                if valid_unwrap {
                                    if let Some(inner_ref_dest) = inner_ref_dest {
                                        if let Some(inner_ref) = inner_ref {
                                            self.mem.cur_frame().store(inner_ref_dest, Value::Ref(inner_ref));
                                        } else {
                                            todo!("tried to get ref to inner value of variant but there was none?");
                                        }
                                    }
                                }
                                self.mem.cur_frame().store(&cond_dest, Value::Bool(valid_unwrap));
                            },
                            _ => bail!("expected ref")
                        }
                    },
                    Instruction::Alloc(dest, r#type) => {
                        let nrf = self.mem.alloc(r#type)?;
                        self.mem.cur_frame().store(dest, nrf);
                    },
                    Instruction::AllocArray(dest, r#type, count) => {
                        let count = match self.mem.cur_frame().convert_value(count) {
                            Value::Int(Integer { signed: false, data, .. }) => data as usize,
                            _ => bail!("invalid count for array alloc")
                        };
                        let nrf = self.mem.alloc_array(r#type, count)?;
                        self.mem.cur_frame().store(dest, nrf);
                    },
                    Instruction::StackAlloc(dest, r#type) => {
                        let nrf = self.mem.stack_alloc(r#type)?;
                        self.mem.cur_frame().store(dest, nrf);
                    },
                    Instruction::StackAllocArray(dest, r#type, count) => {
                        let count = match self.mem.cur_frame().convert_value(count) {
                            Value::Int(Integer { signed: false, data, .. }) => data as usize,
                            _ => bail!("invalid count for array alloc")
                        };
                        let nrf = self.mem.stack_alloc_array(r#type, count)?;
                        self.mem.cur_frame().store(dest, nrf);
                    },

                    Instruction::CopyToStack(dest, src) => {
                        match self.mem.cur_frame().load(src) {
                            Value::Ref(memory::Ref { ty, data }) => {
                                let (copy, size) = if let ir::Type::Array(el_ty) = ty.as_ref() {
                                    let count = unsafe { *(data as *mut usize) };
                                    (self.mem.stack_alloc_array(el_ty.as_ref(), count)?,
                                        self.world.array_size(el_ty, count)?)
                                } else {
                                    (self.mem.stack_alloc(ty.as_ref())?,
                                        self.world.size_of_type(ty.as_ref())?)
                                };
                                if let Value::Ref(copy) = &copy {
                                    unsafe { memcpy(data, copy.data, size); }
                                } else { unreachable!() }
                                self.mem.cur_frame().store(dest, copy);
                            }
                            _ => bail!("expected ref")
                        }
                    },

                    // sad code duplication - should there just be a single alloc function with a
                    // destination argument instead?
                    Instruction::CopyToHeap(dest, src) => {
                        match self.mem.cur_frame().load(src) {
                            Value::Ref(memory::Ref { ty, data }) => {
                                let (copy, size) = if let ir::Type::Array(el_ty) = ty.as_ref() {
                                    let count = unsafe { *(data as *mut usize) };
                                    (self.mem.alloc_array(el_ty.as_ref(), count)?,
                                        self.world.array_size(el_ty, count)?)
                                } else {
                                    (self.mem.alloc(ty.as_ref())?,
                                        self.world.size_of_type(ty.as_ref())?)
                                };
                                if let Value::Ref(copy) = &copy {
                                    unsafe { memcpy(data, copy.data, size); }
                                } else { unreachable!() }
                                self.mem.cur_frame().store(dest, copy);
                            }
                            _ => bail!("expected ref")
                        }
                    },

                    Instruction::SetVariant(dest, new_tag_name, src) => {
                        // this implementation is so unnecessarily ugly. Why? So many nested `match`
                        // expressions! Perhaps we're thinking about this wrong? There's just a lot
                        // to check, and admittedly the other instructions currently do little to
                        // no type checking so maybe this is indicative of spaghetti to follow
                        let src = self.mem.cur_frame().convert_value(src);
                        let src_type = src.type_of(&self.mem);
                        match self.mem.cur_frame().load(dest) {
                            Value::Ref(memory::Ref { ty: dest_ty, data: dest_data }) => {
                                // get the type definition so we can do some type checking and find
                                // the index for name
                                match dest_ty.as_ref() {
                                    ir::Type::User(td_path, None) => {
                                        let td = self.world.get_type(td_path)
                                            .ok_or_else(|| anyhow!("type does not exist"))?;
                                        match td {
                                            ir::TypeDefinition::Sum { variants, .. } => {
                                                // get the index and type definition for the variant named `new_tag_name`
                                                let (variant_index, inner_type_def) =
                                                    variants.iter()
                                                    .enumerate()
                                                    .find(|(_, (n, _ ))| n == new_tag_name)
                                                    .map (|(i, (_, td))| (i, td))
                                                    .ok_or_else(|| anyhow!("variant of sum type does not exist"))?;

                                                // make sure the new data is compatiable with the new tag's inner type definition
                                                match (inner_type_def, &src_type) {
                                                    // no inner data
                                                    (ir::TypeDefinition::Empty, ir::Type::Unit) => {/* do nothing since there is no actual data to copy */},

                                                    // a single inner value
                                                    (ir::TypeDefinition::NewType(inner_t), ir::Type::Ref(refd_src_t)) => {
                                                        if refd_src_t.as_ref() != inner_t {
                                                            bail!("source type {:?} does not match inner destination type {:?}", refd_src_t, inner_type_def);
                                                        }
                                                    },

                                                    // a composite (unnamed) inner value
                                                    (_, ir::Type::Ref(refd_src_t)) => match refd_src_t.as_ref() {
                                                        // make sure that the reference we're copying from has the type of the inner composite
                                                        ir::Type::User(src_typename, _) => {
                                                            if *src_typename != td_path.concat(new_tag_name.clone()) {
                                                                bail!("source type {:?} does not match inner composite destination type {:?}", src_type, inner_type_def);
                                                            }
                                                        },
                                                        t => bail!("expected ref to inner composite type, got {:?} instead", t)
                                                    },

                                                    _ => bail!("type mismatch")
                                                    // ?? need to check if src type === inner_type
                                                    // do we want to only copy out of references,
                                                    // or should we allow copying direct values to
                                                    // allow for literals and fast access to
                                                    // registers -> variant inners???
                                                    // We could interpret Value::Ref srcs as a
                                                    // request to copy the value behind the ref,
                                                    // which will make nested refs inside variants
                                                    // the complex edge case. Perhaps that is
                                                    // ideal, but it makes this Copy... instruction
                                                    // have different semantics than the rest. In
                                                    // all reality this instruction is really more
                                                    // like "seting" or even "initializing" a sum typed value
                                                }

                                                if let Value::Ref(src_ref) = src {
                                                    // write the actual variant tag and copy the data
                                                    unsafe {
                                                        *(dest_data as *mut u64) = variant_index as u64;
                                                        memcpy(src_ref.data, dest_data.offset(std::mem::size_of::<u64>() as isize),
                                                            self.world.size_of_type(&src_type)?);
                                                    }
                                                } else {
                                                    unreachable!()
                                                }
                                            },
                                            _ => bail!("destination not a variant")
                                        }
                                    },
                                    _ => bail!("invalid type for destination reference")
                                }
                            }
                            _ => bail!("expected ref")
                        }
                    }
                }
            }
            prev_block_index = Some(cur_block_index);
            cur_block_index = cur_block.next_block;
        }
    }

}

fn main() {
    env_logger::init();
    let start_mod_path = std::env::args().nth(1).map(ir::Path::from).expect("module path command line argument");
    let start_mod_version = std::env::args().nth(2)
        .map(|vr| ir::VersionReq::parse(&vr).expect("parse starting module version req"))
        .unwrap_or(ir::VersionReq::STAR);
    let mut world = World::new().expect("initialize world");
    world.load_module(&start_mod_path, &start_mod_version).expect("load starting module");
    let mut m = Machine::new(&world);
    m.start(start_mod_path);
}
