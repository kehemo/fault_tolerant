"""
AST representing very simple language involving variables, simple arithmetic operations, and a set of simple commands
(if, while, assignment and sequential composition)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Union, Set, FrozenSet
from dataclasses import dataclass
import re

import cvc5
from cvc5 import Kind


def fresh_var(name):
    counter = getattr(fresh_var, "_counter", 0) + 1
    setattr(fresh_var, "_counter", counter)
    return name + "__" + str(counter)


def get_index_reg_name(reg, idx):
    return f"{reg}_ind_{idx}"


def get_multiplexed_reg_name(reg, idx):
    return f"{reg}_{idx}"


class Expr(ABC):
    @abstractmethod
    def to_cvc5(self, slv, variables, invariants):
        pass

    @abstractmethod
    def variables(self) -> Set[str]:
        pass

    @abstractmethod
    def substitute(self, subst) -> "Expr":
        pass

    @abstractmethod
    def invariant_args(self) -> Dict[str, List[str]]:
        pass

    @abstractmethod
    def stringify(self) -> str:
        pass

    def __and__(self, other: "Expr") -> "Expr":
        return BinOpExpr(Op("and"), self, other)

    def __or__(self, other: "Expr") -> "Expr":
        return BinOpExpr(Op("or"), self, other)

    def __invert__(self) -> "Expr":
        return UnOpExpr(Op("not"), self)

    def implies(self, other: "Expr") -> "Expr":
        return BinOpExpr(Op("implies"), self, other)


@dataclass
class VariableExpr(Expr):
    name: str

    def to_cvc5(self, slv, variables, invariants):
        return variables[self.name]

    def variables(self) -> Set[str]:
        return {self.name}

    def substitute(self, subst) -> "Expr":
        return subst.get(self.name, self)

    def invariant_args(self) -> Dict[str, List[str]]:
        return {}

    def stringify(self) -> str:
        return self.name


@dataclass
class ConstantExpr(Expr):
    value: Union[int, bool]

    def to_cvc5(self, slv, variables, invariants):
        if type(self.value) is int:
            return slv.mkInteger(self.value)
        else:
            return slv.mkBoolean(self.value)

    def variables(self) -> Set[str]:
        return set()

    def substitute(self, subst) -> "Expr":
        return self

    def invariant_args(self) -> Dict[str, List[str]]:
        return {}

    def stringify(self) -> str:
        if type(self.value) is int:
            return str(self.value)
        else:
            return "true" if self.value else "false"


@dataclass
class Op:
    name: str

    def cvc5_kind(self):
        return getattr(Kind, self.name.upper())

    def to_string(self):
        n = self.name.upper()
        if n == "ADD":
            return "+"
        elif n == "SUB":
            return "-"
        elif n == "OR":
            return "||"
        elif n == "AND":
            return "&&"
        elif n == "NOT":
            return "!"
        elif n == "GEQ":
            return ">="
        elif n == "GT":
            return ">"
        elif n == "LT":
            return "<"
        elif n == "LEQ":
            return "<="
        elif n == "EQUAL":
            return "=="
        else:
            assert False


@dataclass
class UnOpExpr(Expr):
    op: Op
    operand: Expr

    def to_cvc5(self, slv, variables, invariants):
        return slv.mkTerm(
            self.op.cvc5_kind(), self.operand.to_cvc5(slv, variables, invariants)
        )

    def variables(self) -> Set[str]:
        return self.operand.variables()

    def substitute(self, subst) -> "Expr":
        return UnOpExpr(self.op, self.operand.substitute(subst))

    def invariant_args(self) -> Dict[str, List[str]]:
        return self.operand.invariant_args()

    def stringify(self) -> str:
        return f"{self.op.to_string()} ({self.operand.stringify()})"


@dataclass
class BinOpExpr(Expr):
    op: Op
    left: Expr
    right: Expr

    def to_cvc5(self, slv, variables, invariants):
        return slv.mkTerm(
            self.op.cvc5_kind(),
            self.left.to_cvc5(slv, variables, invariants),
            self.right.to_cvc5(slv, variables, invariants),
        )

    def variables(self) -> Set[str]:
        return self.left.variables() | self.right.variables()

    def substitute(self, subst) -> "Expr":
        return BinOpExpr(
            self.op, self.left.substitute(subst), self.right.substitute(subst)
        )

    def invariant_args(self) -> Dict[str, List[str]]:
        return {
            **self.left.invariant_args(),
            **self.right.invariant_args(),
        }

    def stringify(self) -> str:
        if self.op.name.upper() == "IMPLIES":
            return f"!({self.left.stringify()}) || ({self.right.stringify()})"
        return f"({self.left.stringify()}) {self.op.to_string()} ({self.right.stringify()})"


@dataclass
class InvariantExpr(Expr):
    invariant_name: str
    args: List[Expr]
    arg_names: List[str]

    def to_cvc5(self, slv, variables, invariants):
        return slv.mkTerm(
            Kind.APPLY_UF,
            invariants[self.invariant_name],
            *[arg.to_cvc5(slv, variables, invariants) for arg in self.args],
        )

    def variables(self) -> Set[str]:
        return set().union(*(arg.variables() for arg in self.args))

    def substitute(self, subst) -> "Expr":
        return InvariantExpr(
            self.invariant_name,
            [arg.substitute(subst) for arg in self.args],
            self.arg_names,
        )

    def invariant_args(self) -> Dict[str, List[str]]:
        return {self.invariant_name: self.arg_names}

    def stringify(self) -> str:
        return (
            f'{self.invariant_name}({",".join([arg.stringify() for arg in self.args])})'
        )


@dataclass
class VerificationContext:
    try_inv: Expr


class Command(ABC):
    @abstractmethod
    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        pass

    @abstractmethod
    def modified_state(self) -> Set[str]:
        pass

    @abstractmethod
    def assigned(self) -> Set[str]:
        pass

    @abstractmethod
    def used_state(self) -> Set[str]:
        pass

    @abstractmethod
    def declared_registers(self) -> Dict[str, Union[None, int]]:
        pass

    @abstractmethod
    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        pass


@dataclass
class SkipCommand(Command):
    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return postcondition

    def modified_state(self) -> Set[str]:
        return set()

    def assigned(self) -> Set[str]:
        return set()

    def registers(self) -> Set[str]:
        return set()

    def used_state(self) -> Set[str]:
        return set()

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        pass


@dataclass
class AssignCommand(Command):
    variable: str
    expression: Expr

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return postcondition.substitute({self.variable: self.expression})

    def modified_state(self) -> Set[str]:
        return {self.variable}

    def assigned(self) -> Set[str]:
        return {self.variable}

    def used_state(self) -> Set[str]:
        return {self.variable} | self.expression.variables()

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        pass


@dataclass
class SetRegisterCommand(Command):
    reg: str
    val: Expr

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return (ctx.try_inv & postcondition).substitute({self.reg: self.val})

    def modified_state(self) -> Set[str]:
        return {self.reg}

    def assigned(self) -> Set[str]:
        return set()

    def used_state(self) -> Set[str]:
        return {self.reg} | self.val.variables()

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        assert self.reg in decl and decl[self.reg] == None


@dataclass
class GetIndirectRegisterCommand(Command):
    reg: str
    offset: Expr
    variable: str
    arity: int = 0

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return postcondition

    def modified_state(self) -> Set[str]:
        return {self.variable}

    def assigned(self) -> Set[str]:
        return {self.variable}

    def used_state(self) -> Set[str]:
        return (
            {f"{get_index_reg_name(self.reg, i)}" for i in range(self.arity)}
            | {f"{get_multiplexed_reg_name(self.reg, i)}" for i in range(self.arity)}
            | self.offset.variables()
        )

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        assert self.reg in decl and decl[self.reg] is not None
        self.arity = decl[self.reg]


@dataclass
class SetIndirectRegisterCommand(Command):
    reg: str
    offset: Expr
    val: Expr
    arity: int = 0

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return postcondition

    def modified_state(self) -> Set[str]:
        return {f"{self.reg}_{i}{tag}" for i in range(self.arity) for tag in [""]}

    def assigned(self) -> Set[str]:
        return set()

    def used_state(self) -> Set[str]:
        return (
            {f"{self.reg}_{i}{tag}" for i in range(self.arity) for tag in ["", "_ind"]}
            | self.offset.variables()
            | self.val.variables()
        )

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        assert self.reg in decl and decl[self.reg] is not None
        self.arity = decl[self.reg]


@dataclass
class IfCommand(Command):
    condition: Expr
    true_command: Command
    false_command: Command

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return (
            self.condition
            & self.true_command.verification_condition(postcondition, ctx)
        ) | (
            (~self.condition)
            & self.false_command.verification_condition(postcondition, ctx)
        )

    def modified_state(self) -> Set[str]:
        return self.true_command.modified_state() | self.false_command.modified_state()

    def assigned(self) -> Set[str]:
        return self.true_command.assigned() | self.false_command.assigned()

    def used_state(self) -> Set[str]:
        return (
            self.true_command.used_state()
            | self.false_command.used_state()
            | self.condition.variables()
        )

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return (
            self.true_command.declared_registers()
            | self.false_command.declared_registers()
        )

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        self.true_command.fill_register_decl(decl)
        self.false_command.fill_register_decl(decl)


@dataclass
class SeqCommand(Command):
    first_command: Command
    second_command: Command

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return self.first_command.verification_condition(
            self.second_command.verification_condition(postcondition, ctx), ctx
        )

    def modified_state(self) -> Set[str]:
        return (
            self.first_command.modified_state() | self.second_command.modified_state()
        )

    def assigned(self) -> Set[str]:
        return self.first_command.assigned() | self.second_command.assigned()

    def used_state(self) -> Set[str]:
        return self.first_command.used_state() | self.second_command.used_state()

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return (
            self.first_command.declared_registers()
            | self.second_command.declared_registers()
        )

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        self.first_command.fill_register_decl(decl)
        self.second_command.fill_register_decl(decl)


@dataclass
class AssertCommand(Command):
    condition: Expr

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return self.condition & postcondition

    def modified_state(self) -> Set[str]:
        return set()

    def assigned(self) -> Set[str]:
        return set()

    def used_state(self) -> Set[str]:
        return self.condition.variables()

    def declared_registers(self) -> Dict[str, Union[int, None]]:
        return {}

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        self.regs = set(decl.keys())


@dataclass
class TryLoopCommand(Command):
    body: Command
    id_mangle: str
    reg_state: FrozenSet[str] = frozenset()

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        used = sorted(list(((self.used_state() | postcondition.variables()) & self.reg_state)))
        universal_regs = {
            reg: VariableExpr(fresh_var(reg))
            for reg in self.modified_state() & self.reg_state
        }
        invariant_id = fresh_var("inv")
        inv = InvariantExpr(invariant_id, [VariableExpr(reg) for reg in used], used)
        return (
            inv &
            inv.implies(
                self.body.verification_condition(
                    postcondition, VerificationContext(inv)
                )
            ).substitute(universal_regs)
        )

    def modified_state(self) -> Set[str]:
        return {
            x if x in self.reg_state else f"{x}_{self.id_mangle}"
            for x in self.body.modified_state()
        }

    def assigned(self) -> Set[str]:
        return self.body.assigned()

    def used_state(self) -> Set[str]:
        return {
            x if x in self.reg_state else f"{x}_{self.id_mangle}"
            for x in self.body.used_state()
        }

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return self.body.declared_registers()

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        self.body.fill_register_decl(decl)
        self.reg_state = frozenset(
            name
            for k, v in decl.items()
            for name in (
                [k]
                if v is None
                else [get_index_reg_name(k, idx) for idx in range(v)]
                + [get_multiplexed_reg_name(k, idx) for idx in range(v)]
            )
        )


@dataclass
class TryCatchCommand(Command):
    try_body: Command
    catch_body: Command
    id_mangle: str
    reg_state: FrozenSet[str] = frozenset()

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        used = sorted(list(((self.used_state() | postcondition.variables()) & self.reg_state)))
        invariant_id = fresh_var("inv")
        universal_regs = {
            reg: VariableExpr(fresh_var(reg))
            for reg in set(used) & self.modified_state()
        }
        inv = InvariantExpr(invariant_id, [VariableExpr(reg) for reg in used], used)
        return (inv &
            self.try_body.verification_condition(
                postcondition, VerificationContext(inv)
            )
            & inv.implies(
                self.catch_body.verification_condition(postcondition, ctx)
            ).substitute(universal_regs)
        )

    def modified_state(self) -> Set[str]:
        return self.try_body.modified_state() | {
            x if x in self.reg_state else f"{x}_{self.id_mangle}"
            for x in self.catch_body.modified_state()
        }

    def assigned(self) -> Set[str]:
        return self.try_body.assigned() | self.catch_body.assigned()

    def used_state(self) -> Set[str]:
        return self.try_body.used_state() | {
            x if x in self.reg_state else f"{x}_{self.id_mangle}"
            for x in self.catch_body.modified_state()
        }

    def declared_registers(self) -> Dict[str, Union[int, None]]:
        return self.try_body.declared_registers() | self.catch_body.declared_registers()

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        self.try_body.fill_register_decl(decl)
        self.catch_body.fill_register_decl(decl)
        self.reg_state = frozenset(
            name
            for k, v in decl.items()
            for name in (
                [k]
                if v is None
                else [get_index_reg_name(k, idx) for idx in range(v)]
                + [get_multiplexed_reg_name(k, idx) for idx in range(v)]
            )
        )


@dataclass
class DeclareRegistersCommand(Command):
    regs: Dict[str, Union[None, int]]

    def verification_condition(
        self, postcondition: Expr, ctx: VerificationContext
    ) -> Expr:
        return postcondition

    def modified_state(self) -> Set[str]:
        return set()

    def assigned(self) -> Set[str]:
        return set()

    def used_state(self) -> Set[str]:
        return set()

    def declared_registers(self) -> Dict[str, Union[None, int]]:
        return self.regs

    def fill_register_decl(self, decl: Dict[str, Union[None, int]]):
        assert self.regs.items() <= decl.items()


# LANGUAGE PARSER BEGIN
NAME_GUARD = r"(?![a-zA-Z0-9_])"

TOKEN_SPECIFICATION = [
    ("NEWLINE", r"\n"),
    ("WHITESPACE", r"[\t ]+"),
    ("SEMICOLON", r";"),
    ("COMMA", r","),
    ("COLON", r":"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LARROW", r"<-"),
    ("RARROW", r"->"),
    ("LBRACKET", r"\["),
    ("RBRACKET", r"\]"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("GEQ", r">="),
    ("GT", r">"),
    ("EQUAL", r"=="),
    ("NEQ", r"!="),
    ("LEQ", r"<="),
    ("LT", r"<"),
    ("OR", r"\|\|"),
    ("AND", r"&&"),
    ("NOT", r"!"),
    ("ADD", r"\+"),
    ("SUB", r"\-"),
    ("MUL", r"\*"),
    ("DIV", r"/"),
    ("ASSIGN", r"="),
    ("REG", r"reg" + NAME_GUARD),
    ("ASSERT", r"assert" + NAME_GUARD),
    ("TRY", r"try" + NAME_GUARD),
    ("CATCH", r"catch" + NAME_GUARD),
    ("TRYLOOP", r"tryloop" + NAME_GUARD),
    ("IF", r"if" + NAME_GUARD),
    ("ELSE", r"else" + NAME_GUARD),
    ("NAME", r"[a-zA-Z][a-zA-Z0-9_]*"),
    ("NUMBER", r"(0|[1-9][0-9]*)"),
    ("ERROR", r"."),
]


@dataclass
class Token:
    kind: str  # Token type (e.g. 'LCURLY', 'VAR', 'INT')
    value: str = None  # Token value (e.g. '{', 'x', '45')
    line: int = -1  # Line number (default = -1)
    column: int = -1  # Column number (default = -1)


class TokenStream:
    def __init__(self, code):
        """
        Tokenizes `code` into a list of tokens. Tokens are matched
        according to `TOKEN_SPECIFICATION`. If more than one regular expression
        matches, then the first match will be preferred.
        """
        self.tokens = []
        self.token_index = 0
        pos = 0
        line_num = 1
        line_start = 0
        while True:
            for kind, regex in TOKEN_SPECIFICATION:
                m = re.compile(regex).match(code, pos)
                if m is not None:
                    break

            if m is None:
                self.tokens.append(Token("EOF", "", line_num, column))
                break

            pos = m.end()
            value = m.group()
            column = m.start() - line_start
            if kind == "NEWLINE":
                line_start = m.end()
                line_num += 1
            elif kind == "ERROR":
                raise RuntimeError(
                    f"{value!r} unexpected at line:{line_num} col:{column}"
                )
            elif kind != "WHITESPACE":
                self.tokens.append(Token(kind, value, line_num, column))

    def peek(self):
        return (
            self.tokens[self.token_index]
            if self.token_index < len(self.tokens)
            else None
        )

    def consume(self):
        self.token_index += 1

    def seek(self, pos):
        self.token_index = pos

    def get_index(self):
        return self.token_index


def seq(*parsers):
    def combined_parser(stream):
        start_pos = stream.get_index()
        res = []
        for parser in parsers:
            parser_res = parser(stream)
            if parser_res is None:
                res = None
                stream.seek(start_pos)
                break
            else:
                res += parser_res
        return res

    return combined_parser


def alt(*parsers):
    def combined_parser(stream):
        for parser in parsers:
            parser_res = parser(stream)
            if parser_res is not None:
                return parser_res
        return None

    return combined_parser


def kleene(parser):
    def new_parser(stream):
        res = []
        while True:
            parser_res = parser(stream)
            if parser_res is None:
                break
            else:
                res += parser_res
        return res

    return new_parser


def option(parser):
    def new_parser(stream):
        parser_res = parser(stream)
        return [] if parser_res is None else parser_res

    return new_parser


def lit(token_kind):
    def new_parser(stream):
        next_token = stream.peek()
        if next_token.kind == token_kind:
            stream.consume()
            return [] if token_kind == "EOF" else [next_token]

    return new_parser


def parse_program(stream):
    res = seq(parse_command, lit("EOF"))(stream)
    if res is not None:
        prog = res[0]
        decl = prog.declared_registers()
        assert len(prog.assigned() & decl.keys()) == 0
        prog.fill_register_decl(decl)
        return [prog]
    else:
        return None


def parse_command(stream):
    coms = kleene(
        alt(
            parse_assign_command,
            parse_set_register_command,
            parse_get_ind_register_command,
            parse_set_ind_register_command,
            parse_if_command,
            parse_tryloop_command,
            parse_assert_command,
            parse_try_catch_command,
            parse_declare_registers_command,
        )
    )(stream)
    cur = SkipCommand()
    for prev in reversed(coms):
        cur = SeqCommand(prev, cur)
    return [cur]


def parse_assign_command(stream):
    res = seq(lit("NAME"), lit("ASSIGN"), parse_int, lit("SEMICOLON"))(stream)
    if res is not None:
        return [AssignCommand(res[0].value, res[2])]
    else:
        return None


def parse_set_register_command(stream):
    res = seq(lit("NAME"), lit("LARROW"), parse_int, lit("SEMICOLON"))(stream)
    if res is not None:
        return [SetRegisterCommand(res[0].value, res[2])]
    else:
        return None


def parse_get_ind_register_command(stream):
    res = seq(
        lit("NAME"),
        lit("LBRACKET"),
        parse_int,
        lit("RBRACKET"),
        lit("RARROW"),
        lit("NAME"),
        lit("SEMICOLON"),
    )(stream)
    if res is not None:
        return [GetIndirectRegisterCommand(res[0].value, res[2], res[5].value)]
    else:
        return None


def parse_set_ind_register_command(stream):
    res = seq(
        lit("NAME"),
        lit("LBRACKET"),
        parse_int,
        lit("RBRACKET"),
        lit("LARROW"),
        parse_int,
        lit("SEMICOLON"),
    )(stream)
    if res is not None:
        return [SetIndirectRegisterCommand(res[0].value, res[2], res[5])]
    else:
        return None


def parse_if_command(stream):
    res = seq(
        lit("IF"),
        parse_bool,
        lit("LBRACE"),
        parse_command,
        lit("RBRACE"),
        option(seq(lit("ELSE"), lit("LBRACE"), parse_command, lit("RBRACE"))),
    )(stream)
    if res is not None:
        return [IfCommand(res[1], res[3], res[7] if len(res) > 7 else SkipCommand())]
    else:
        return None


def parse_tryloop_command(stream):
    res = seq(lit("TRYLOOP"), lit("LBRACE"), parse_command, lit("RBRACE"))(stream)
    if res is not None:
        return [TryLoopCommand(res[2], fresh_var("mangle"))]
    else:
        return None


def parse_assert_command(stream):
    res = seq(lit("ASSERT"), parse_bool, lit("SEMICOLON"))(stream)
    if res is not None:
        return [AssertCommand(res[1])]
    else:
        return None


def parse_try_catch_command(stream):
    res = seq(
        lit("TRY"),
        lit("LBRACE"),
        parse_command,
        lit("RBRACE"),
        lit("CATCH"),
        lit("LBRACE"),
        parse_command,
        lit("RBRACE"),
    )(stream)
    if res is not None:
        return [TryCatchCommand(res[2], res[6], fresh_var("mangle"))]
    else:
        return None


def parse_declare_registers_command(stream):
    res = seq(
        lit("REG"),
        kleene(
            seq(lit("NAME"), option(seq(lit("COLON"), lit("NUMBER"))), lit("COMMA"))
        ),
        lit("NAME"),
        option(seq(lit("COLON"), lit("NUMBER"))),
        lit("SEMICOLON"),
    )(stream)
    if res is not None:
        regs = {}
        cur = 1
        while cur < len(res):
            assert res[cur].kind == "NAME"
            if res[cur + 1].kind == "COLON":
                assert res[cur + 2].kind == "NUMBER"
                regs[res[cur].value] = int(res[cur + 2].value)
                cur += 4
            else:
                regs[res[cur].value] = None
                cur += 2
        return [DeclareRegistersCommand(regs)]
    else:
        return None


def parse_bool(stream):
    res = seq(parse_conj, kleene(seq(lit("OR"), parse_conj)))(stream)
    if res is not None:
        cur = res[0]
        for tok in res[2::2]:
            cur = BinOpExpr(Op("or"), cur, tok)
        return [cur]
    else:
        return None


def parse_conj(stream):
    res = seq(parse_bool_unit, kleene(seq(lit("AND"), parse_bool_unit)))(stream)
    if res is not None:
        cur = res[0]
        for tok in res[2::2]:
            cur = BinOpExpr(Op("and"), cur, tok)
        return [cur]
    else:
        return None


def parse_bool_unit(stream):
    res = alt(
        seq(
            parse_int,
            alt(lit("GEQ"), lit("GT"), lit("EQUAL"), lit("NEQ"), lit("LEQ"), lit("LT")),
            parse_int,
        ),
        seq(lit("NOT"), parse_bool),
        seq(lit("LPAREN"), parse_bool, lit("RPAREN")),
    )(stream)
    if res is not None:
        if type(res[0]) == Token:
            if res[0].kind == "NOT":
                return [UnOpExpr(Op("not"), res[1])]
            else:
                return [res[1]]
        else:
            if res[1].kind == "NEQ":
                return [UnOpExpr(Op("not"), BinOpExpr(Op("equal"), res[0], res[2]))]
            else:
                return [BinOpExpr(Op(res[1].kind), res[0], res[2])]
    else:
        return None


def parse_int(stream):
    res = seq(parse_prod, kleene(seq(alt(lit("ADD"), lit("SUB")), parse_prod)))(stream)
    if res is not None:
        cur_term = res[0]
        cur = 2
        while cur < len(res):
            cur_term = BinOpExpr(Op(res[cur - 1].kind), cur_term, res[cur])
            cur += 2
        return [cur_term]
    else:
        return None


def parse_var_expr(stream):
    res = lit("NAME")(stream)
    if res is not None:
        return [VariableExpr(res[0].value)]
    else:
        return None


def parse_const_expr(stream):
    res = lit("NUMBER")(stream)
    if res is not None:
        return [ConstantExpr(int(res[0].value))]
    else:
        return None


def parse_prod(stream):
    res = seq(parse_unit, kleene(seq(alt(lit("MUL"), lit("DIV")), parse_unit)))(stream)
    if res is not None:
        cur_term = res[0]
        cur = 2
        while cur < len(res):
            cur_term = BinOpExpr(
                Op("MULT" if res[cur - 1].kind == "MUL" else "INTS_DIVISION"),
                cur_term,
                res[cur],
            )
            cur += 2
        return [cur_term]
    else:
        return None


def parse_unit(stream):
    res = alt(
        parse_var_expr, parse_const_expr, seq(lit("LPAREN"), parse_int, lit("RPAREN"))
    )(stream)
    if res is not None:
        if len(res) == 1:
            return [res[0]]
        else:
            return [res[1]]
    else:
        return None


# LANGUAGE PARSER END


def boolean_grammar(slv, variables):
    # You should copy and adapt the example from the grammar
    integer = slv.getIntegerSort()
    boolean = slv.getBooleanSort()

    # declare input variables for the functions-to-synthesize
    slv_vars = [slv.mkVar(integer, v) for v in variables]

    # declare the grammar non-terminals
    start = slv.mkVar(integer, "StartInt")
    start_bool = slv.mkVar(boolean, "Start")

    # define the rules
    zero = slv.mkInteger(0)
    one = slv.mkInteger(1)

    # Kinds are
    # Kinds are listed here https://cvc5.github.io/docs/cvc5-1.0.2/api/python/base/kind.html
    plus = slv.mkTerm(Kind.ADD, start, start)
    minus = slv.mkTerm(Kind.SUB, start, start)
    # times = slv.mkTerm(Kind.MULT, start, start)
    # ite = slv.mkTerm(Kind.ITE, start_bool, start, start)

    Or = slv.mkTerm(Kind.OR, start_bool, start_bool)
    And = slv.mkTerm(Kind.AND, start_bool, start_bool)
    Not = slv.mkTerm(Kind.NOT, start_bool)
    leq = slv.mkTerm(Kind.LEQ, start, start)
    lt = slv.mkTerm(Kind.LT, start, start)
    eq = slv.mkTerm(Kind.EQUAL, start, start)

    # create the grammar object
    g = slv.mkGrammar(slv_vars, [start_bool, start])

    # bind each non-terminal to its rules
    g.addRules(start, [zero, one, *slv_vars])  # , plus, minus])
    g.addRules(start_bool, [Or, And, Not, eq, leq, lt])

    return g, slv_vars


def solve(assumption, vc):
    slv = cvc5.Solver()

    # required options
    # slv.setOption("verbose", "true")
    slv.setOption("sygus", "true")
    slv.setOption("strict-parsing", "true")
    slv.setOption("incremental", "false")

    # set the logic
    # slv.setLogic("LIA")
    slv.setLogic("ALL")

    invariant_args = vc.invariant_args()
    all_arg_names = list(set((arg for args in invariant_args.values() for arg in args)))

    g, arg_vars = boolean_grammar(slv, all_arg_names)
    integer = slv.getIntegerSort()
    boolean = slv.getBooleanSort()

    arg_dict = {arg_name: arg for arg_name, arg in zip(all_arg_names, arg_vars)}
    invariants = {}
    for inv_name, args in invariant_args.items():
        print(f"(synth-fun {inv_name} ({' '.join([f'({arg} Int)' for arg in args])}) Bool {g})")
        invariants[inv_name] = slv.synthFun(
            inv_name, [arg_dict[arg] for arg in args], boolean, g
        )
    variables = {}
    for variable in vc.variables() | assumption.variables():
        variables[variable] = slv.declareSygusVar(variable, integer)
        print(f"(declare-var {variable} Int)")

    pre_f = assumption.to_cvc5(slv, variables, invariants)
    # print(pre_f)
    post_f = vc.to_cvc5(slv, variables, invariants)
    # print(post_f)
    print(f"(constraint {slv.mkTerm(Kind.IMPLIES, pre_f, post_f)})")
    print("(check-synth)")
    slv.addSygusConstraint(
        slv.mkTerm(Kind.IMPLIES, pre_f, post_f),
    )
    if slv.checkSynth().hasSolution():
        return slv.getSynthSolutions(list(invariants.values()))
    return "could not solve"


def atomic_write():
    program = """
reg a, b, dirty;
tryloop {
    a <- 0;
    b <- 0;
    dirty <- 0;
}
try {
    dirty <- 1;
    a <- c;
    b <- d;
    assert a == c && b == d;
} catch {
    if dirty == 1 {
        tryloop {
            a <- 0;
            b <- 0;
        }
    }
}
assert (a == c && b == d) || (a == 0 && b == 0);
"""
    example = parse_program(TokenStream(program))[0]
    precondition = ConstantExpr(True)
    postcondition = ConstantExpr(True)
    return dict(
        example=example,
        precondition=precondition,
        postcondition=postcondition,
    )


def transfer():
    program = """
reg a, b, t, dirty, x, y, n;
tryloop {
    dirty <- 0;
}
try {
    t <- a + b;
    dirty <- 1;
    a <- a - n;
    b <- b + n;
} catch {
    if (dirty == 1) {
        tryloop {
            b <- t - a;
        }
    }
}
assert a == x - n && b == y + n || a == x && b == y;
"""
    example = parse_program(TokenStream(program))[0]
    precondition = parse_bool(TokenStream("a == x && b == y"))[0]
    postcondition = ConstantExpr(True)
    return dict(
        example=example,
        precondition=precondition,
        postcondition=postcondition,
    )

def swap_reg():
    program = """
reg r1, r2, dirty, temp, a, b;
tryloop {
    r1 <- a;
    r2 <- b;
    dirty <- 0;
}
try {
    temp <- r1;
    dirty <- 1;
    r1 <- r2;
    r2 <- temp;
    assert r1 == b && r2 == a;
} catch {
    if dirty == 1 && r2 != temp {
        tryloop {
            r1 <- temp;
        }
    }
}
assert r1 == a && r2 == b || r1 == b && r2 == a;
    """
    example = parse_program(TokenStream(program))[0]
    precondition = ConstantExpr(True)
    postcondition = ConstantExpr(True)
    return dict(
        example=example,
        precondition=precondition,
        postcondition=postcondition,
    )


examples = {
    "aw": atomic_write,
    "tx": transfer,
    "sr": swap_reg,
}


def main():
    from sys import argv

    if len(argv) == 1 or argv[1] not in examples:
        print(f"Usage: python3 fallible_parser.py <{'|'.join(examples.keys())}>")
        exit(1)
    eg = examples[argv[1]]()
    print("AST:")
    print(eg["example"])
    vc = eg["example"].verification_condition(
        eg["postcondition"], VerificationContext(eg["postcondition"])
    )
    print("Precond:")
    print(eg["precondition"].stringify())
    print("VC:")
    print(vc.stringify())
    cond = eg["precondition"].implies(vc)
    print("Invariant declarations:")
    for inv_name, args in cond.invariant_args().items():
        print(
            f"bit {inv_name}({','.join(f'int {arg}' for arg in args)})"
            + "{return exp_bool(BOUND, {"
            + ",".join(args)
            + "}, {0, 1});}"
        )
    print("Assertion:")
    print(
        f"bit check({','.join(f'int {var}' for var in sorted(cond.variables()))}) "
        + "{return"
        + cond.stringify()
        + ";}"
    )
    print("Solving:")
    print(solve(eg["precondition"], vc))


if __name__ == "__main__":
    main()
