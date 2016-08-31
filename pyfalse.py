"""
False 1.2b.

TODO: Read a .f file and execute the False program as a command-line app.

    python3.5 -m false somefile.f

TODO: Use Unicode.  http://aplwiki.com/UnicodeAplTable


"""
from collections.abc import Iterator
import sys
import io
import logging


class Peekable(Iterator):
    """Looks ahead one character. Also has a final sentinel behavior of returning None."""

    def __init__(self, code):
        super().__init__()
        self.code = code
        self.index = 0

    def peek(self):
        if self.index != len(self.code):
            return self.code[self.index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.index != len(self.code):
            c = self.code[self.index]
            self.index += 1
            return c
        return None


class Token:
    """
    False Token superclass. Different subclasses have different features.
    """

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def evaluate(self, falseMachine):
        """
        Many tokens simply push themselves onto the stack.
        :param falseMachine:
        """
        self.logger.debug('push {}'.format(self))
        falseMachine.stack.append(self)


class Lambda(Token, list):
    """
    Block of code. Executed by the :class:`EvalOperator`.
    """

    def __init__(self, tokens=[]):
        super().__init__()
        if tokens: self.extend(tokens)

    @property
    def text(self):
        return ''.join(self)

    def __repr__(self):
        return '[' + ' '.join(repr(c) for c in self) + ']'


class VarRef(Token):
    """
    Reference to a variable. Originally, these might have been
    memory addresses. Now they're simple names with no value.

    Set with :class:`SetOperator`. Get with :class:`GetOperator`.
    """

    def __init__(self, name):
        super().__init__()
        self.name = name

    def __repr__(self):
        return self.name


class Value(Token):
    """
    Integer value.
    """

    def __init__(self, value):
        super().__init__()
        self.value = value

    def __repr__(self):
        return repr(self.value)


class Operator(Token):
    """
    Superclass for operators.
    """

    def __init__(self, oper):
        super().__init__()
        self.oper = oper

    def __repr__(self):
        return self.oper

    def evaluate(self, falseMachine):
        raise NotImplementedError


class SetOperator(Operator):
    """
    pop (value, varadr) -

    Updates variable state
    """
    def evaluate(self, falseMachine):
        variable = falseMachine.stack.pop(-1)
        value = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {},{}'.format(self.oper, variable, value))
        falseMachine.variables[variable.name] = value


class GetOperator(Operator):
    """
    pop (varadr) - push (value of variable)
    """
    def evaluate(self, falseMachine):
        variable = falseMachine.stack.pop(-1)
        value = falseMachine.variables[variable.name]
        self.logger.debug('{}:pop {} - push {}'.format(self.oper, variable, value))
        falseMachine.stack.append(value)


class EvalOperator(Operator):
    """
    pop (function) - push (evaluation of function)

    Evaluates function.
    """
    def evaluate(self, falseMachine):
        lambda_object = falseMachine.stack.pop(-1)
        self.logger.debug('{}:eval {}'.format(self.oper, lambda_object))
        falseMachine.run(lambda_object)


class AddOperator(Operator):
    """
    pop (n1, n2) - push (n1+n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value + n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class SubOperator(Operator):
    """
    pop (n1, n2) - push (n1-n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value - n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class MulOperator(Operator):
    """
    pop (n1, n2) - push (n1×n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value * n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class DivOperator(Operator):
    """
    pop (n1, n2) - push (n1÷n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value // n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class NegOperator(Operator):
    """
    pop (n1) - push (-n1)
    """
    def evaluate(self, falseMachine):
        n1 = falseMachine.stack.pop(-1)
        value = Value(-n1.value)
        self.logger.debug('{}:pop {} - push {}'.format(self.oper, n1, value))
        falseMachine.stack.append(value)


class EqOperator(Operator):
    """
    pop (n1, n2) - push (n1 == n2)

    True == -1, False == 0
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(-1 if n1.value == n2.value else 0)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class GtOperator(Operator):
    """
    pop (n1, n2) - push (n1 > n2)

    True == -1, False == 0
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(-1 if n1.value > n2.value else 0)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class AndOperator(Operator):
    """
    pop (n1, n2) - push (n1 & n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value & n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class OrOperator(Operator):
    """
    pop (n1, n2) - push (n1 | n2)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        value = Value(n1.value | n2.value)
        self.logger.debug('{}:pop {} {} - push {}'.format(self.oper, n1, n2, value))
        falseMachine.stack.append(value)


class NotOperator(Operator):
    """
    pop (n1) - push (~n1)
    """
    def evaluate(self, falseMachine):
        n1 = falseMachine.stack.pop(-1)
        value = Value(~ n1.value)
        self.logger.debug('{}:pop {} - push {}'.format(self.oper, n1, value))
        falseMachine.stack.append(value)


class DupOperator(Operator):
    """
    pop (n) - push (n, n)
    """
    def evaluate(self, falseMachine):
        n1 = falseMachine.stack.pop(-1)
        self.logger.debug('{}: - push {} {}'.format(self.oper, n1, n1))
        falseMachine.stack.append(n1)
        falseMachine.stack.append(n1)


class DropOperator(Operator):
    """
    pop (n) -
    """
    def evaluate(self, falseMachine):
        n1 = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {} - '.format(self.oper, n1))


class SwapOperator(Operator):
    """
    pop (n1, n2) - push (n2, n1)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {} {} - push {} {}'.format(self.oper, n1, n2, n2, n1))
        falseMachine.stack.append(n2)
        falseMachine.stack.append(n1)


class RotOperator(Operator):
    """
    pop (n0, n1, n2) - push (n1, n2, n0)
    """
    def evaluate(self, falseMachine):
        n2 = falseMachine.stack.pop(-1)
        n1 = falseMachine.stack.pop(-1)
        n = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {} {} {} - push {} {} {}'.format(self.oper, n, n1, n2, n1, n2, n))
        falseMachine.stack.append(n1)
        falseMachine.stack.append(n2)
        falseMachine.stack.append(n)


class PickOperator(Operator):
    """
    pop (m) - push (stack[n])
    """
    def evaluate(self, falseMachine):
        n = falseMachine.stack.pop(-1).value
        value = falseMachine.stack[-(n + 1)]
        self.logger.debug('{}: - push {}'.format(self.oper, value))
        falseMachine.stack.append(value)


class IfOperator(Operator):
    """
    pop (v, fun) -

    If v is True, evaluate fun.
    """
    def evaluate(self, falseMachine):
        fun = falseMachine.stack.pop(-1)
        bool_value = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {} {}'.format(self.oper, bool_value, fun))
        if bool_value.value:
            falseMachine.run(fun)


class WhileOperator(Operator):
    """
    pop (while_fun, fun) -

    Evaluate while_fun
    while True:
        Evaluate fun
        Evaluate while_fun
    """
    def evaluate(self, falseMachine):
        fun = falseMachine.stack.pop(-1)
        boolf = falseMachine.stack.pop(-1)
        self.logger.debug('{}:pop {} {}'.format(self.oper, boolf, fun))
        falseMachine.run(boolf)
        while falseMachine.stack.pop(-1).value:
            falseMachine.run(fun)
            falseMachine.run(boolf)


class PrintNumOperator(Operator):
    """
    pop (n) -

    Format and write the number on sys.stdout
    """
    def evaluate(self, falseMachine):
        n = falseMachine.stack.pop(-1)
        print(n, file=falseMachine.output, end='')


class PutcOperator(Operator):
    """
    pop (ch) -

    write the Unicode character on sys.stdout
    """
    def evaluate(self, falseMachine):
        n = falseMachine.stack.pop(-1)
        print(chr(n.value), file=falseMachine.output, end='')


class GetcOperator(Operator):
    """
    get (ch) -

    get one Unicode character from sys.stdin
    """
    def evaluate(self, falseMachine):
        char = falseMachine.input.read(1)
        falseMachine.stack.append(Value(ord(char)))


class FlushOperator(Operator):
    """
    Flush stdout. While possible helpful for some kinds of interactive
    programs, it's only necessary when switching from Putc to Getc.
    """
    def evaluate(self, falseMachine):
        falseMachine.output.flush()


class Comment(Token):
    """
    The {} token which is equivalent to whitespace.
    This has no effect on the stack.
    """

    def __init__(self, text):
        super().__init__()
        self.text = text

    def __repr__(self):
        return repr(self.text)

    def evaluate(self, falseMachine):
        pass


class StringOutput(Token):
    """
    The "quote" operator which emits a string.
    This has no effect on the stack.
    """

    def __init__(self, text):
        super().__init__()
        self.text = text

    def __repr__(self):
        return '"{}"'.format(self.text)

    def evaluate(self, falseMachine):
        self.logger.debug('print {!r}'.format(self.text))
        print(self.text, file=falseMachine.output, end='')


class Unsupported(Exception):
    pass


class ASCIIParser:
    """Parse a block of text into a sequence of False Tokens.

    This is the 1.2b Extended ASCII (Latin-1, ISO/IEC 8859-1).
    Requires two non-ASCII characters, ø and ß.

    Also works with Windows code page 1252.

    Whitespace is generally insignificant, but not always.
    The '<char> syntax, for instance, does not ignore whitespace.
    """
    OPERATORS = {
        ':': SetOperator,
        ';': GetOperator,
        '!': EvalOperator,

        '+': AddOperator,
        '-': SubOperator,
        '*': MulOperator,
        '/': DivOperator,
        '_': NegOperator,

        '=': EqOperator,
        '>': GtOperator,

        '&': AndOperator,
        '|': OrOperator,
        '~': NotOperator,

        '$': DupOperator,
        '%': DropOperator,
        '\\': SwapOperator,
        '@': RotOperator,

        'ø': PickOperator,

        '?': IfOperator,
        '#': WhileOperator,

        '.': PrintNumOperator,
        ',': PutcOperator,
        '^': GetcOperator,
        'ß': FlushOperator,
    }

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def consume(self, peekable, until='}'):
        """
        Consume characters from the current peekable until the target character.

        :param peekable: Iterator over input characters
        :param until: End of input, default is '}' to help parse comments.
        :yields: individual characters
        """
        c = next(peekable)
        while c is not None and c != until:
            yield c
            c = next(peekable)
        yield c

    def parse(self, peekable, until=None):
        """
        Parse the source text, yielding Token objects.

        :param peekable: Iterator over input characters
        :param until: End of input, default is None; use ']' to parse lambdas.
        :yields: tokens.
        """
        for c in peekable:
            if c is None or c == until: break
            if c.isspace(): continue

            if c == '{':
                text = ['{'] + list(self.consume(peekable))
                yield Comment(''.join(text))

            elif c == '[':
                lambda_tokens = self.parse(peekable, until=']')
                yield Lambda(lambda_tokens)

            elif c in self.OPERATORS:
                # Note two alpha in these single-char tokens.
                oper_class = self.OPERATORS[c]
                yield oper_class(c)

            elif c.isalpha(): # and c not in 'øß':
                yield VarRef(c)

            elif c.isdigit():
                number = int(c)
                while peekable.peek().isdigit():
                    c = next(peekable)
                    number = 10 * number + int(c)
                yield Value(number)

            elif c == "'":
                char = next(peekable)
                yield Value(ord(char))

            elif c == "`":
                # Value on top of stack is a number; this is 68000 machine code.
                raise Unsupported("`")

            elif c == '"':
                text = ""
                c = next(peekable)
                while c != '"':
                    text += c
                    c = next(peekable)
                yield StringOutput(text)

            else:
                raise Exception("Unexpected Character")


class UnicodeParser(ASCIIParser):
    """Parse a block of text into a sequence of False Tokens.

    This contains the Unicode alternative symbols in addition
    to the original ASCII symbols.

    Whitespace is generally insignificant, but not always.
    The '<char> syntax, for instance, does not ignore whitespace.
    """
    OPERATORS = {
        ':': SetOperator,   # U+003A
        '←': SetOperator,   # U+2190  2a←
        ';': GetOperator,   # U+003B
        '→': GetOperator,   # U+2192  a→
        '!': EvalOperator,  # U+0021
        '⍎': EvalOperator,  # U+234E  2[1+]⍎

        '+': AddOperator,   # U+002B  2 3+
        '-': SubOperator,   # U+002D  5 7-
        '*': MulOperator,   # U+002A
        '×': MulOperator,   # U+00D7  11 13×
        '/': DivOperator,   # U+002F
        '÷': DivOperator,   # U+00F7  17 2÷
        '_': NegOperator,   # U+005F
        '¯': NegOperator,   # U+00AF  19¯

        '=': EqOperator,    # U+003D  2 3 =~
        '>': GtOperator,    # U+003E  5 7 >

        '&': AndOperator,   # U+0026
        '∧': AndOperator,   # U+2227  1 5∧
        '|': OrOperator,    # U+007C
        '∨': OrOperator,    # U+2228  1 2∨
        '~': NotOperator,   # U+007E  1 ~
        '∼': NotOperator,   # U+223C  1 ∼

        '$': DupOperator,   # U+0024
        '↑': DupOperator,   # U+2191  ↑
        '%': DropOperator,  # U+0025
        '↓': DropOperator,  # U+2193  ↓
        '\\': SwapOperator, # U+005C
        '⌽': SwapOperator,  # U+233D  ⌽
        '@': RotOperator,   # U+0040
        '⍉': RotOperator,   # U+2349  ⍉

        'ø': PickOperator,  # U+00F8
        '⊃': PickOperator,  # U+2283  ⊃

        '?': IfOperator,    # U+003F
        '#': WhileOperator, # U+0023
        '⍟': WhileOperator, # U+235F  ⍟

        '.': PrintNumOperator, # U+002E
        '⍕': PrintNumOperator, # U+2355
        ',': PutcOperator,  # U+002C
        '⍈': PutcOperator,  # U+2348
        '^': GetcOperator,  # U+005E
        '⍇': GetcOperator,  # U+2347
        'ß': FlushOperator, # U+00DF
        '⌺': FlushOperator, # U+233A ⌺
    }

class FalseMachine:
    """
    The False VM. Executes a sequence of tokens.

    In the case of an overall script, a single Lambda is built
    from the whole sequence.
    """

    def __init__(self, input=sys.stdin, output=sys.stdout):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.input = input
        self.output = output
        self.stack = []
        self.variables = {}

    def run(self, lambda_code):
        for token in lambda_code:
            token.evaluate(self)


def getmachine(source, input='', *, parser=UnicodeParser()):
    """
    Parse code, return the final VM so that we can examine the state.

    :param source: Code to parse and run.
    :param input: Input stream
    :return: FalseMachine instance after running the code.
    """
    code = Lambda(parser.parse(Peekable(source)))
    input = io.StringIO(input)
    output = io.StringIO()
    machine = FalseMachine(input, output)
    machine.run(code)
    return machine


def run(source, input='', *, parser=UnicodeParser()):
    """
    Parse the code, return the final output.

    :param source: Code to parse and run.
    :param input: Input stream.
    :return: Output
    """
    code = Lambda(parser.parse(Peekable(source)))
    input = io.StringIO(input)
    output = io.StringIO()
    machine = FalseMachine(input, output)
    machine.run(code)
    return machine.output.getvalue()


__test__ = {
'comment': '''
>>> source = '{ this is a comment }'
>>> fm = getmachine(source)
>>> fm.stack
[]
''',

'lambda': '''
>>> source = '2[1+]!'
>>> machine = getmachine(source)
>>> machine.stack
[3]
''',

'lambda 2': '''
>>> source = '[1+]i:  2i;!'
>>> machine = getmachine(source)
>>> machine.stack
[3]
''',

'variables': '''
>>> source = '1a:     a;1+b:   b;'
>>> machine = getmachine(source)
>>> machine.stack
[2]
''',

'stack $': '''
>>> machine = getmachine('1$')
>>> machine.stack
[1, 1]
''',

'stack %': '''
>>> machine = getmachine('1 2%')
>>> machine.stack
[1]
''',

'stack \\': '''
>>> machine = getmachine('1 2\\\\')
>>> machine.stack
[2, 1]
''',

'stack @': '''
>>> machine = getmachine('1 2 3@')
>>> machine.stack
[2, 3, 1]
''',

'stack ø': '''
>>> machine = getmachine('7 8 9 2ø')
>>> machine.stack
[7, 8, 9, 7]
''',

'control ? 1': '''
>>> machine = getmachine('1a:  a;1=["hello!"]?')
>>> machine.output.getvalue()
'hello!'
>>> machine.stack
[]
''',

'control ? 2': '''
>>> machine = getmachine('1a:  a;1=$["true"]?~["false"]?')
>>> machine.output.getvalue()
'true'
>>> machine.stack
[]
''',

'control #': '''
>>> machine = getmachine("3a:  [a;0=~][a;.' , a;1-a:]#")
>>> machine.output.getvalue()
'3 2 1 '
>>> machine.stack
[]
''',

'arithmethic': '''
>>> machine = getmachine('1 2 + 2 3 - 5 7 * 31 11 / 13_')
>>> machine.stack
[3, -1, 35, 2, -13]
''',

'comparison and logic': '''
>>> machine = getmachine('1 2=~ 2 3> 1 3& 1 2| 0~')
>>> machine.stack
[-1, 0, 1, 3, -1]
''',

'comparison and logic 2': '''
>>> machine = getmachine('2a:  a;0>a;99>~&')
>>> machine.stack
[-1]
>>> machine = getmachine('101a:  a;0>a;99>~&')
>>> machine.stack
[0]
''',

'demo1': """
>>> code = '''
... "Hello, Word!
... "
... '''
>>> run(code)
'Hello, Word!\\n'
""",

# This one is awkwardly complex because of the \\ escapes nested two deep.
'demo2': """
>>> code = '''
... { writes all prime numbers between 0 and 100 }
...
... 99 9[1-$][\\\\$@$@$@$@\\\\/*=[1-$$[%\\\\1-$@]?0=[\\\\$.' ,\\\\]?]?]#
... '''
>>> run(code)
'97 89 83 79 73 71 67 61 59 53 47 43 41 37 31 29 23 19 17 13 11 7 5 3 2 '
""",

# Use Unicode, it's nicer.
'demo2u': """
>>> code = '''
... { writes all prime numbers between 0 and 100 }
...
... 99 9[1-↑][⌽↑⍉↑⍉↑⍉↑⍉⌽÷×=[1-↑↑[↓⌽1-↑@]?0=[⌽↑.' ,⌽]?]?]⍟
... '''
>>> run(code)
'97 89 83 79 73 71 67 61 59 53 47 43 41 37 31 29 23 19 17 13 11 7 5 3 2 '
""",


# Another demo
'demo3': """
>>> code = '''
... {factorial program in false!}
...
... [$1 = ~[$1 - f;! *]?]f:          {fac() in false}
...
... "calculate the factorial of [1..8]: "
... ß ^ ß
... '0-$$0>~\\\\8>|$
... "result: "
... ~[\\\\f;!.]?
... ["illegal input!"]?"
... "
... '''
>>> run(code, input='5')
'calculate the factorial of [1..8]: result: 120\\n'
""",

'demo3u': """
>>> code = '''
... {factorial program in false!}
...
... [↑1 = ~[↑1 - f→⍎ ×]?]f←          {fac() in false}
...
... "calculate the factorial of [1..8]: "
... ⌺ ⍇ ⌺
... '0-↑↑0>~⌽8>|↑
... "result: "
... ~[⌽f→⍎.]?
... ["illegal input!"]?"
... "
... '''
>>> run(code, input='6')
'calculate the factorial of [1..8]: result: 720\\n'
"""

}


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=0)
