Module(
    path: Path([Symbol("basic_sum")]),
    version: "0.0.1",
    types: {
        Symbol("foo"): Sum(
            parameters: [],
            variants: [
                (Symbol("Nothing"), Empty),
                (Symbol("Something"), NewType(Int(signed: false, width: 64))),
                (Symbol("Pair"), Product(
                    parameters: [],
                    fields: [
                        (Symbol("first"), Int(signed: false, width: 64)),
                        (Symbol("second"), Int(signed: false, width: 64))
                    ]
                ))
            ]
        )
    },
    interfaces: {},
    implementations: {},
    functions: {
        Symbol("start"): (
            FunctionSignature(args: [], return_type: Int(width: 64, signed: false)),
            FnBody(
                max_registers: 10,
                blocks: [
                    BasicBlock(
                        instrs: [
                            Alloc(Register(0), User(Path([Symbol("basic_sum"), Symbol("foo")]), None)),
                            SetVariant(Register(0), Symbol("Nothing"), LiteralUnit),
                            UnwrapVariant(Register(1), None, Register(0), Symbol("Nothing")),
                            Br(cond: Reg(Register(1)), if_true: 1, if_false: 999),
                        ],
                        next_block: 999
                    ),
                    BasicBlock(
                        instrs: [
                            Alloc(Register(2), Int(signed: false, width: 64)),
                            StoreRef(Register(2), LiteralInt(Integer(width: 64, signed: false, data: 18328))),
                            SetVariant(Register(0), Symbol("Something"), Reg(Register(2))),
                            UnwrapVariant(Register(1), Some(Register(3)), Register(0), Symbol("Something")),
                            LoadRef(Register(5), Register(3)), // important to load here, because this reference points to the inside of the variant, which is about to get modified
                            Br(cond: Reg(Register(1)), if_true: 2, if_false: 999),
                        ],
                        next_block: 999
                    ),
                    BasicBlock(
                        instrs: [
                            Alloc(Register(2), User(Path([Symbol("basic_sum"), Symbol("foo"), Symbol("Pair")]), None)),
                            StoreField(LiteralInt(Integer(width: 64, signed: false, data: 8383)), Register(2), Symbol("first")),
                            StoreField(LiteralInt(Integer(width: 64, signed: false, data: 9945)), Register(2), Symbol("second")),
                            SetVariant(Register(0), Symbol("Pair"), Reg(Register(2))),
                            UnwrapVariant(Register(1), Some(Register(4)), Register(0), Symbol("Pair")),
                            Br(cond: Reg(Register(1)), if_true: 3, if_false: 999),
                        ],
                        next_block: 999
                    ),
                    BasicBlock(
                        instrs: [
                            //this is not good example code! it is unsafe to use unwrap variant to get two differently typed references to the same place in memory!
                            LoadField(Register(6), Register(4), Symbol("first")),
                            LoadField(Register(7), Register(4), Symbol("second")),
                            BinaryOp(Add, Register(8), Reg(Register(6)), Reg(Register(7))),
                            BinaryOp(Sub, Register(9), Reg(Register(8)), Reg(Register(5))),
                            Return(Reg(Register(9)))
                        ],
                        next_block: 999
                    )
                ]
            )
        )
    },
    imports: []
)
