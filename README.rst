False
#####

See http://strlen.com/false-language

See http://strlen.com/false/false.txt

Here's a couple of example programs to give a taste of what False looks like:

Copy Files

::

    { copy.f: copy file. usage: copy < infile > outfile  }

    ß[^$1_=~][,]#


Factorial

::

    { factorial program in false! }

    [$1=~[$1-f;!*]?]f:          { fac() in false }

    "calculate the factorial of [1..8]: "
    ß^ß'0-$$0>~\8>|$
    "result: "
    ~[\f;!.]?
    ["illegal input!"]?"
    "

Prime Numbers

::

    { writes all prime numbers between 0 and 100 }

    99 9[1-$][\$@$@$@$@\/*=[1-$$[%\1-$@]?0=[\$.' ,\]?]?]#

The point is ... well ... the point is hard to articulate. But there it is.
False implemented in Python.

In order to make things microscopically easier to read, I've supplemented
the original ASCII-based symbols with Unicode.

::

    { writes all prime numbers between 0 and 100 }

    99 9[1-↑][⌽↑⍉↑⍉↑⍉↑⍉⌽÷×=[1-↑↑[↓⌽1-↑@]?0=[⌽↑.' ,⌽]?]?]⍟

::

    {factorial program in false!}

    [↑1 = ~[↑1 - f→⍎ ×]?]f←          {fac() in false}

    "calculate the factorial of [1..8]: "
    ⌺ ⍇ ⌺
    '0-↑↑0>~⌽8>|↑
    "result: "
    ~[⌽f→⍎.]?
    ["illegal input!"]?"
    "

Yes. I borrowed a few APL operators because they seemed more meaningful than
the original ASCII.
