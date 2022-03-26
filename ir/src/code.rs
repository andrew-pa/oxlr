//! This module defines the structure of the IR virtual machine code, which is in single static
//! assignment form. Code is always found inside function bodies as [`FnBody`], organized into
//! single [`BasicBlock`]s, each of which represents a continuous path of execution.
use std::{borrow::Cow, collections::HashMap};

use serde::{Serialize, Deserialize};
use super::{Symbol, Path, Type, numbers::Integer, numbers::Float};

/// A reference to a virtual machine register
/// In keeping with SSA form, a register can only be assigned to once in the program, but the value can be used many times
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub struct Register(pub u32);

/// A reference to a [`BasicBlock`] within a function body, by index
pub type BlockIndex = usize;

/// A value that is stored in the IR code
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Value {
    /// The literal unit value `()`
    LiteralUnit,
    /// A literal integer value of arbitrary width and sign
    LiteralInt(Integer),
    /// A literal floating point value
    LiteralFloat(Float),
    /// A string literal
    LiteralString(String),
    /// A boolean literal
    LiteralBool(bool),
    /// A reference to the value stored in a register
    Reg(Register)
}

impl Value {
    fn unwrap_reg(self) -> Register {
        match self {
            Value::Reg(r) => r,
            _ => panic!("attempted to unwrap value ({:?}) as a register", self)
        }
    }
}

/// Operations on two values that can be executed by the [`BinaryOp`](Instruction::BinaryOp) instruction.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum BinOp {
    Add, Sub, Mul, Div,
    Shl, Shr,
    LAnd, LOr, Eq, NEq, Less, Greater, LessEq, GreaterEq
}

/// Operations on a single value that can be executed by the [`UnaryOp`](Instruction::UnaryOp) instruction.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum UnaryOp {
    LogNot, BitNot, Neg
}


/// A single virtual machine instruction.
/// Destination registers are typically first in the tuple, then the source value
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Instruction {
    /// A SSA Phi node, which selects a value to store in its destination register depending on
    /// which basic block jumped to this instruction
    Phi(
        /// Destination register
        Register,
        /// A mapping from origin blocks to values
        std::collections::HashMap<BlockIndex, Value>
    ),

    /// Jump to a different block based on the value of `cond`
    Br {
        cond: Value,
        /// Index of the block to jump to if `cond` is true
        if_true: BlockIndex,
        /// Index of the block to jump to if `cond` is false
        if_false: BlockIndex
    },

    /// Compute a binary operation from [`BinOp`] on two values, putting the result in the
    /// destination register
    BinaryOp(
        /// Operation to perform
        BinOp,
        /// Destination register
        Register,
        /// First input value
        Value,
        /// Second input value
        Value
    ),
    /// Compute a unary operation from [`UnaryOp`] on a values, putting the result in the
    /// destination register
    UnaryOp(
        /// Operation to perform
        UnaryOp,
        /// Destination register
        Register,
        /// Input value
        Value
    ),

    /// Copy a value into a register directly
    LoadImm(Register, Value),

    /// Get the value behind a reference on the heap
    LoadRef(
        /// Destination register
        Register,
        /// Register containing the source reference
        Register
    ),
    /// Move a value into a reference on the heap
    StoreRef(
        /// Register containing the destination reference
        Register,
        /// Value to store behind reference
        Value
    ),

    /// Compute the reference to a index into a reference to an array or tuple
    RefIndex(
        /// Destination register for reference to inner data
        Register,
        /// Source container reference
        Register,
        /// Index
        Value
    ),

    /// Compute the reference to a field within a structure
    RefField(
        /// Destination register for reference to inner field
        Register,
        /// Source structure reference
        Register,
        /// Field name
        Symbol
    ),

    /// Loads the indexed value starting from zero in the referenced array or tuple on the heap
    LoadIndex(
        /// Destination register
        Register,
        /// Source array/tuple reference
        Register,
        /// Index
        Value
    ),

    /// Stores a value at an index into an array or tuple on the heap
    StoreIndex(
        /// Source reference
        Register,
        /// Index
        Value,
        /// Value to store behind reference
        Value
    ),

    /// Load a value in a field in a structure referenced on the heap
    LoadField(
        /// Destination register
        Register,
        /// Source reference
        Register,
        /// Field name
        Symbol
    ),
    /// Store a value in a field in a structure referenced on the heap
    StoreField(
        /// Value to store in field
        Value,
        /// Register that contains the reference to the destination
        Register,
        /// Field name
        Symbol
    ),

    /// Call a function referenced by the path, placing the return value in the destination register
    Call(
        /// Destination for return value
        Option<Register>,
        /// Path to the function
        Path,
        /// Argument values
        Vec<Value>
    ),
    /// Call the implementation function for the specified interface function, placing the return
    /// value in the destination register. The first parameter's type will be used to find the specific implementation
    CallImpl(
        /// Destination for return value
        Option<Register>,
        /// Path to the function on the interface (not the implementation)
        Path,
        /// Argument values
        Vec<Value>
    ),
    /// Return from this function, yielding specified value
    Return(Value),

    /// Create a function pointer to a function at the path
    RefFunc(Register, Path),

    /// Test to see if a sum type value matches a specific variant, optionally unwrapping its contained value and putting a reference to it in a register
    UnwrapVariant(
        /// Destination register set to true or false depending on if this was a successful match 
        Register,
        /// Optional destination register store the reference to the inner value of the variant
        Option<Register>,
        /// The variant value to test
        Register,
        /// The name of the variant to test for
        Symbol
    ),

    /// Allocate a value on the heap of a specified type and put a reference in the destination register
    Alloc(Register, Type),

    /// Allocate an array of values on the heap and put an array value reference in the destination register.
    AllocArray(
        /// Destination register
        Register,
        /// Element type
        Type,
        /// Number of elements in the array
        Value
    ),

    /// Allocate a value on the stack of a specified type and put a reference in the destination register
    /// The value will be destroyed when the function returns, rendering the reference invalid
    StackAlloc(Register, Type),

    /// Allocate an array of values on the stack and put an array value reference in the destination register.
    /// The array will be destroyed when the function returns, rendering the reference invalid
    StackAllocArray(
        /// Destination register
        Register,
        /// Element type
        Type,
        /// Number of elements in the array
        Value
    ),

    /// Copy data out of a reference onto the stack, making a new stack allocation. Performs a shallow copy
    CopyToStack(
        /// Destination register for stack reference
        Register,
        /// Reference to copy to the stack
        Register
    ),

    /// Copy data out of a reference onto the heap, making a new heap allocation. Performs a shallow copy
    CopyToHeap(
        /// Destination register for heap reference
        Register,
        /// Reference to copy into the heap
        Register
    ),

    /// Copy data out of a reference into a sum data object, changing the variant tag in the process and replacing whatever was inside before
    SetVariant(
        /// Reference to destination sum data object
        Register,
        /// Name of variant to tag data with
        Symbol,
        /// Source reference to copy or nil for selecting a variant that doesn't have inner data
        Value,
    )
}

/// A basic block of continuous execution within a function body. Execution proceeds sequentially from the first instruction in `instrs`.
/// If execution comes to the end of the block, the block indexed by `next_block` will be executed.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BasicBlock {
    pub instrs: Vec<Instruction>,
    pub next_block: BlockIndex
}

/// The actual code for a function that will be executed when it is called
/// Execution begins at block number 0
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FnBody {
    /// The maximum number of registers that will be used by the function body
    pub max_registers: u32,
    /// The basic blocks that are contained in the function body. [`BlockIndex`] values are indices inside this vector.
    pub blocks: Vec<BasicBlock>
}

pub struct FnBodyBuilder {
    res: FnBody
}

pub struct BasicBlockBuilder<'parent> {
    fnbody: &'parent mut FnBodyBuilder,
    block_index: usize
}

impl FnBodyBuilder {
    pub fn new() -> FnBodyBuilder {
        FnBodyBuilder {
            res: FnBody {
                max_registers: 0,
                blocks: Vec::new()
            }
        }
    }

    /// Start adding a new basic block to the function
    pub fn start_block(&mut self) -> BasicBlockBuilder {
        let block_index = self.res.blocks.len();

        self.res.blocks.push(BasicBlock {
            instrs: Vec::new(),
            next_block: usize::MAX
        });

        BasicBlockBuilder {
            fnbody: self,
            block_index
        }
    }

    pub fn next_register(&mut self) -> Register {
        let reg = Register(self.res.max_registers);
        self.res.max_registers += 1;
        reg
    }

    pub fn build(self) -> FnBody { self.res }
}

impl<'a> BasicBlockBuilder<'a> {
    /// Finish buliding the block, returning the index of the new block
    pub fn build(self) -> BlockIndex {
        self.block_index
    }

    fn push_instr(&mut self, instr: Instruction) {
        self.fnbody.res.blocks[self.block_index].instrs.push(instr);
    }

    fn push_instr_with_dest(&mut self, f: impl FnOnce(Register)->Instruction) -> Value {
        let dest = self.fnbody.next_register();
        self.push_instr(f(dest));
        Value::Reg(dest)
    }

    pub fn phi(&mut self, values: HashMap<BlockIndex, Value>) -> Value {
        self.push_instr_with_dest(|dest| Instruction::Phi(dest, values))
    }

    pub fn branch(&mut self, cond: Value, if_true: BlockIndex, if_false: BlockIndex) {
        self.push_instr(Instruction::Br { cond, if_true, if_false });
    }

    pub fn binary_op(&mut self, op: BinOp, left: Value, right: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::BinaryOp(op, dest, left, right))
    }

    pub fn unary_op(&mut self, op: UnaryOp, val: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::UnaryOp(op, dest, val))
    }

    pub fn load_immediate(&mut self, val: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::LoadImm(dest, val))
    }

    pub fn load_ref(&mut self, r#ref:  Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::LoadRef(dest, r#ref.unwrap_reg()))
    }

    pub fn store_ref(&mut self, dest: Value, value: Value) {
        self.push_instr(Instruction::StoreRef(dest.unwrap_reg(), value));
    }

    pub fn ref_index(&mut self, cont: Value, index: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::RefIndex(dest, cont.unwrap_reg(), index))
    }

    pub fn ref_field(&mut self, cont: Value, field_name: Symbol) -> Value {
        self.push_instr_with_dest(|dest| Instruction::RefField(dest, cont.unwrap_reg(), field_name))
    }

    pub fn call(&mut self, fn_path: Path, args: Vec<Value>) -> Value {
        self.push_instr_with_dest(|dest| Instruction::Call(Some(dest), fn_path, args))
    }

    pub fn call_ignore_ret(&mut self, fn_path: Path, args: Vec<Value>) {
        self.push_instr(Instruction::Call(None, fn_path, args))
    }

    pub fn call_impl(&mut self, interface_fn_path: Path, args: Vec<Value>) -> Value {
        self.push_instr_with_dest(|dest| Instruction::CallImpl(Some(dest), interface_fn_path, args))
    }

    pub fn call_impl_ignore_ret(&mut self, interface_fn_path: Path, args: Vec<Value>) {
        self.push_instr(Instruction::CallImpl(None, interface_fn_path, args))
    }

    pub fn ret(&mut self, val: Value) {
        self.push_instr(Instruction::Return(val));
    }

    pub fn check_variant(&mut self, val: Value, variant_name: Symbol) -> Value {
        self.push_instr_with_dest(|dest| Instruction::UnwrapVariant(dest, None, val.unwrap_reg(), variant_name))
    }

    pub fn unwrap_variant(&mut self, val: Value, variant_name: Symbol) -> (Value, Value) {
        let ref_dest = self.fnbody.next_register();
        (self.push_instr_with_dest(|dest| Instruction::UnwrapVariant(dest, Some(ref_dest), val.unwrap_reg(), variant_name)), Value::Reg(ref_dest))
    }

    pub fn alloc(&mut self, ty: Type) -> Value {
        self.push_instr_with_dest(|dest| Instruction::Alloc(dest, ty))
    }

    pub fn alloc_array(&mut self, ty: Type, size: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::AllocArray(dest, ty, size))
    }

    pub fn stack_alloc(&mut self, ty: Type) -> Value {
        self.push_instr_with_dest(|dest| Instruction::StackAlloc(dest, ty))
    }

    pub fn stack_alloc_array(&mut self, ty: Type, size: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::StackAllocArray(dest, ty, size))
    }

    pub fn copy_to_stack(&mut self, r#ref: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::CopyToStack(dest, r#ref.unwrap_reg()))
    }

    pub fn copy_to_heap(&mut self, r#ref: Value) -> Value {
        self.push_instr_with_dest(|dest| Instruction::CopyToHeap(dest, r#ref.unwrap_reg()))
    }

    pub fn set_variant(&mut self, new_tag_name: Symbol, new_value: Option<Value>) -> Value {
        self.push_instr_with_dest(|dest| Instruction::SetVariant(dest, new_tag_name, 
                new_value.unwrap_or(Value::LiteralUnit)))
    }
}

