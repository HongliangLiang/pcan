from typing import List

import pandas as pd
# A simple visitor for FuncDef nodes that prints the names and
# locations of function definitions.
from pycparser.c_ast import *

logic = {'Compound', 'DoWhile', 'FileAST', 'For', 'FuncDef', 'If', 'While', 'Switch', 'Case', 'Default', 'Label'}
num = {"0", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"}
uop = {'++', '--', 'p++', 'p--'}

CHILD = 0
PARENT = 1
DATA = 2
SUBTOKEN = 3

color = ['blue', 'red']


def unwanted(node: Node):
    return node.__class__.__name__ in logic


class BasicVisitor:
    _method_cache = None

    def __init__(self):
        self.cnt = 0
        self.edges = {}
        self.nodes = {}
        self.pointers = set()

    def json(self):
        x, mask = list(zip(*[i.values() for i in self.nodes.values()]))
        edges = tuple(self.edges.keys())
        etype = tuple(i['label'] for i in self.edges.values())
        return pd.Series([x, mask, edges, etype])

    def build_graph(self):
        builder = []
        for e in self.edges.keys():
            label = self.edges[e]['label']
            builder.append(f'{e[0]}->{e[1]} [label="{label}"];')

        for n in self.nodes.keys():
            c = color[self.nodes[n].get('mask')]

            label = self.nodes[n]['label']
            builder.append(f'{n}[label="{label}",color={c}];')

        body = '\n'.join(builder)
        prop = 'node [shape = box, style="filled,solid"];'
        return f"digraph ProgramGraph {{\n{prop}\n{body} \n}}"

    def visit(self, node, left=False):
        """ Visit a node.
        """

        if self._method_cache is None:
            self._method_cache = {}

        visitor = self._method_cache.get(node.__class__.__name__, None)
        if visitor is None:
            method = 'visit_' + node.__class__.__name__
            visitor = getattr(self, method, self.generic_visit)
            self._method_cache[node.__class__.__name__] = visitor

        return visitor(node, left)

    def generic_visit(self, node, left=False):
        """ Called if no explicit visitor function exists for a
            node. Implements preorder visiting of the node.
        """
        for c in node:
            self.visit(c, left)

    def next_node(self, label):

        n = self.cnt
        self.add_node(n, f'{label}')
        self.cnt += 1
        return n

    def split_name(self, n):
        # parts = split_identifier_into_parts(n)
        # parts = [i for i in parts if not re.match(r'\d+', i)]
        #
        # dummy = self.next_node('<pad>')
        # for sub in parts:
        #     n = self.next_node(sub)
        #     self.wrapper(dummy, [n, ], SUBTOKEN)

        return self.next_node(n)

    def helper_visit(self, n, read=None, write=None, left=False):

        idx, r, w = self.visit(n, left)
        if read is not None:
            read.extend(r)
        if write is not None:
            write.extend(w)

        return idx

    def add_node(self, n, label):

        self.nodes[n] = {"label": label, "mask": 0}

    def add_edge(self, parent, child, label=0):
        self.edges[(parent, child)] = {"label": label}

        if self.nodes[parent]['label'] in logic and \
                self.nodes[child]['label'] not in logic:
            self.nodes[child]['mask'] = 1

    def wrapper(self, node: int, body: List, label=PARENT):
        for c in body:
            self.add_edge(node, c, label)


class GeneralVisitor(BasicVisitor):

    def visit_ArrayDecl(self, node: ArrayDecl, left):
        root = self.next_node("ArrayDecl")

        if node.type is not None:
            t = self.helper_visit(node.type)
            self.wrapper(root, [t, ])
        if node.dim is not None:
            d = self.helper_visit(node.dim)
            self.wrapper(root, [d, ])

        return root, [], []

    def visit_ArrayRef(self, node: ArrayRef, left):

        read, write = [], []

        root = self.next_node("ArrayRef")

        if node.name is not None:
            n = self.helper_visit(node.name, read, write, left)
            self.wrapper(root, [n])

        if node.subscript is not None:
            s = self.helper_visit(node.subscript, read, write, left=False)
            self.wrapper(root, [s])

        return root, read, write

    def visit_Assignment(self, node: Assignment, left):
        # print(node.op)

        read, write = [], []
        l = self.helper_visit(node.lvalue, read, write, left=True)
        r = self.helper_visit(node.rvalue, read, write)

        root = self.next_node(node.op)
        self.wrapper(root, [l, r])

        return root, read, write

    def visit_Constant(self, node: Constant, left):

        value = ""
        if node.type == 'string':
            value = 'StringLiteral'
        elif node.type == 'int':
            if node.value not in num:
                value = "<NUM>"
            else:
                value = node.value
        else:

            if node.value == "' '":
                value = "<blank>"
            else:
                value = node.value

        return self.next_node(value), [], []

    def visit_BinaryOp(self, node: BinaryOp, left):
        read, write = [], []
        root = self.next_node(node.op)

        if node.left is not None:
            l = self.helper_visit(node.left, read, write, left)
            self.wrapper(root, [l, ])

        if node.right is not None:
            r = self.helper_visit(node.right, read, write, left)

            self.wrapper(root, [r, ])
        return root, read, write

    def visit_Break(self, node: Break, left):
        return self.next_node("Break"), [], []

    def visit_Case(self, node: Case, left):
        s = [self.helper_visit(node.expr)]
        for i in (node.stmts or []):
            s.append(self.helper_visit(i))

        root = self.next_node("Case")
        self.wrapper(root, s, CHILD)
        return root, [], []

    def visit_Cast(self, node: Cast, left):
        root = self.next_node("Cast")
        read, write = [], []

        e = self.helper_visit(node.expr, read, write)
        t = self.helper_visit(node.to_type)
        self.wrapper(root, [t, e])
        return root, read, write

    def visit_Compound(self, node: Compound, left):
        items = []
        statements = []

        for item in (node.block_items or []):
            idx, r, w = self.visit(item)
            if unwanted(item):
                statements.append('X')
            else:
                statements.append((idx, r, w))
            items.append(idx)

        for i in range(0, len(statements)):
            if statements[i] == 'X':
                continue
            write = set(statements[i][2])
            for w in write:
                for j in reversed(range(0, i)):
                    if statements[j] == 'X':
                        break
                    if w in statements[j][2]:
                        self.add_edge(statements[j][0], statements[i][0], 2)
                        break
                    elif w in statements[j][1]:
                        self.add_edge(statements[j][0], statements[i][0], 2)

            read = set(statements[i][1])
            for r in read:
                for j in reversed(range(0, i)):
                    if statements[j] == 'X':
                        break
                    if r in statements[j][2]:
                        self.add_edge(statements[j][0], statements[i][0], 2)
                        break

        root = self.next_node("Compound")
        self.wrapper(root, items, CHILD)
        return root, [], []

    def visit_Decl(self, node: Decl, left):
        root = self.next_node("Decl")

        read, write = [], []

        if node.type is not None:
            t, _, _ = self.visit(node.type)
            if node.type.__class__.__name__ in ['PtrDecl', 'ArrayDecl']:
                self.pointers.add(node.name)

            self.wrapper(root, [t, ])
        if node.name is not None:
            n = self.split_name(node.name)
            self.wrapper(root, [n, ])
        write.append(node.name)

        if node.init is not None:
            i = self.helper_visit(node.init, read, write, left=False)
            self.wrapper(root, [i, ])

        return root, read, write

    def visit_TypeDecl(self, node: TypeDecl, left):

        return self.helper_visit(node.type), [], []

    def visit_DeclList(self, node: DeclList, left):
        # print(node.__class__.__name__)
        read, write = [], []
        decls = []
        for d in node.decls:
            decls.append(self.helper_visit(d, read, write))

        root = self.next_node("DeclList")
        self.wrapper(root, decls)
        return root, read, write

    def visit_Default(self, node: Default, left):

        s = []
        for i in (node.stmts or []):
            s.append(self.helper_visit(i))

        root = self.next_node("Default")
        self.wrapper(root, s, CHILD)
        return root, [], []

    def visit_DoWhile(self, node: DoWhile, left):
        root = self.next_node("DoWhile")

        if node.cond is not None:
            c = self.helper_visit(node.cond)
            self.wrapper(root, [c, ], CHILD)
        if node.stmt is not None:
            s = self.helper_visit(node.stmt)
            self.wrapper(root, [s, ], CHILD)

        return root, [], []

    def visit_EllipsisParam(self, node: EllipsisParam, left):
        pass

    def visit_ParamList(self, node: ParamList, left):
        ext = []
        for i in node.params:
            ext.append(self.helper_visit(i))
        root = self.next_node("ParamList")
        self.wrapper(root, ext)
        return root, [], []

    def visit_EmptyStatement(self, node: EmptyStatement, left):
        return self.next_node("EmptyStatement"), [], []

    def visit_ExprList(self, node: ExprList, left):
        ext = []
        read, write = [], []
        for expr in node.exprs:
            ext.append(self.helper_visit(expr, read, write))

        root = self.next_node("ExprList")
        self.wrapper(root, ext)
        return root, read, write

    def visit_FileAST(self, node: FileAST, left):
        # print(node.__class__.__name__)
        ext = []
        for i in node:
            ext.append(self.helper_visit(i))

        root = self.next_node("FileAST")
        self.wrapper(root, ext, CHILD)
        return root, [], []

    def visit_For(self, node: For, left):
        root = self.next_node("For")

        if node.init is not None:
            i = self.helper_visit(node.init)
            self.wrapper(root, [i, ], CHILD)

        if node.cond is not None:
            c = self.helper_visit(node.cond)
            self.wrapper(root, [c, ], CHILD)
        if node.next is not None:
            n = self.helper_visit(node.next)
            self.wrapper(root, [n, ], CHILD)

        if node.stmt is not None:
            s = self.helper_visit(node.stmt)
            self.wrapper(root, [s, ], CHILD)

        return root, [], []

    def visit_FuncCall(self, node: FuncCall, left):
        root = self.next_node("FuncCall")
        read, write = [], []
        if node.name is not None:
            name = self.helper_visit(node.name)
            self.wrapper(root, [name, ])
        if node.args is not None:
            args, r, w = self.visit(node.args)
            for x in r:
                if x in self.pointers:
                    write.append(x)
                else:
                    read.append(x)
            write.extend(w)

            self.wrapper(root, [args, ])

        return root, read, write

    def visit_FuncDef(self, node: FuncDef, left):
        # print(node.__class__.__name__)
        root = self.next_node("FuncDef")

        if node.decl is not None:
            decl = self.helper_visit(node.decl)
            self.wrapper(root, [decl, ], CHILD)

        if node.body is not None:
            body = self.helper_visit(node.body)
            self.wrapper(root, [body, ], CHILD)

        return root, [], []

    def visit_FuncDecl(self, node: FuncDecl, left):
        # print(node.__class__.__name__)
        root = self.next_node("FuncDecl")

        t = self.helper_visit(node.type)
        self.wrapper(root, [t, ])

        if node.args is not None:
            args = self.helper_visit(node.args)
            self.wrapper(root, [args, ])

        return root, [], []

    def visit_Goto(self, node: Goto, left):
        root = self.next_node("Goto")
        c = self.next_node(node.name)

        self.wrapper(root, [c, ])
        return root, [], []

    def visit_ID(self, node: ID, left):
        dummy = self.split_name(node.name)

        if left:
            return dummy, [], [node.name]

        return dummy, [node.name], []

    def visit_IdentifierType(self, node: IdentifierType, left):
        root = self.next_node("IdentifierType")
        for sub in node.names:
            n = self.next_node(sub)
            self.wrapper(root, [n, ], SUBTOKEN)

        return root, [], []

    def visit_If(self, node: If, left):
        root = self.next_node("If")
        cond = self.helper_visit(node.cond)
        iftrue = self.helper_visit(node.iftrue)
        self.wrapper(root, [cond, iftrue], CHILD)
        if node.iffalse is not None:
            iffalse = self.helper_visit(node.iffalse)
            self.wrapper(root, [iffalse], CHILD)

        return root, [], []

    def visit_InitList(self, node: InitList, left):
        ext = []
        for i in node.exprs:
            ext.append(self.helper_visit(i))

        root = self.next_node("InitList")
        self.wrapper(root, ext)
        return root, [], []

    def visit_Label(self, node: Label, left):
        root = self.next_node("Label")

        if node.stmt is not None:
            stmt = self.helper_visit(node.stmt)
            self.wrapper(root, [stmt, ], CHILD)

        return root, [], []

    def visit_NamedInitializer(self, node: NamedInitializer, left):
        pass

    def visit_PtrDecl(self, node: PtrDecl, left):
        root = self.next_node("PtrDecl")
        self.wrapper(root, [self.helper_visit(node.type), ])

        return root, [], []

    def visit_Return(self, node: Return, left):
        root = self.next_node("Return")
        read, write = [], []

        if node.expr is not None:
            expr = self.helper_visit(node.expr, read, write)
            self.wrapper(root, [expr, ])

        return root, read, write

    def visit_Struct(self, node: Struct, left):
        root = self.next_node("Struct")
        ext = []

        for i in (node.decls or []):
            ext.append(self.helper_visit(i))

        self.wrapper(root, ext)

        if node.name is not None:
            n = self.split_name(node.name)
            self.wrapper(root, [n, ])

        return root, [], []

    def visit_StructRef(self, node: StructRef, left):
        root = self.next_node("StructRef")
        read, write = [], []
        if node.name is not None:
            n = self.helper_visit(node.name, read, write, left)

            self.wrapper(root, [n, ])
        if node.field is not None:
            f = self.helper_visit(node.field)
            self.wrapper(root, [f, ])

        return root, read, write

    def visit_Switch(self, node: Switch, left):
        root = self.next_node("Switch")
        self.wrapper(root, [self.helper_visit(node.cond), self.helper_visit(node.stmt)], CHILD)
        return root, [], []

    def visit_TernaryOp(self, node: TernaryOp, left):
        root = self.next_node("TernaryOp")
        self.wrapper(root,
                     [self.helper_visit(node.cond),
                      self.helper_visit(node.iftrue),
                      self.helper_visit(node.iffalse)])
        return root, [], []

    def visit_Typedef(self, node: Typedef, left):
        root = self.next_node("Typedef")
        self.wrapper(root, [self.helper_visit(node.type), self.split_name(node.name)])
        return root, [], []

    def visit_Typename(self, node: Typename, left):
        return self.helper_visit(node.type), [], []

    def visit_UnaryOp(self, node: UnaryOp, left):
        root = self.next_node(node.op)
        read, write = [], []

        eid, r, w = self.visit(node.expr, left)

        if node.op in uop:
            if node.expr.__class__.__name__ == 'ID':
                read.append(node.expr.name)
                write.append(node.expr.name)
            else:
                read.extend(r)
                write.extend(w)
                if node.expr.__class__.__name__ == 'ArrayRef' and \
                        node.expr.name.__class__.__name__ == 'ID':
                    write.append(node.expr.name.name)
        else:
            read.extend(r)
            write.extend(w)

        self.wrapper(root, [eid, ])
        return root, read, write

    def visit_Union(self, node: Union, left):
        pass

    def visit_While(self, node: While, left):
        root = self.next_node("While")

        if node.cond is not None:
            cond = self.helper_visit(node.cond)
            self.wrapper(root, [cond, ], CHILD)
        if node.stmt is not None:
            body = self.helper_visit(node.stmt)
            self.wrapper(root, [body, ], CHILD)
        return root, [], []

    def visit_Continue(self, node: Continue, left):
        return self.next_node("Continue"), [], []

    def visit_Enum(self, node: Enum, left):
        root = self.next_node("Enum")
        if node.name is not None:
            n = self.split_name(node.name)
            self.wrapper(root, [n, ])
        self.wrapper(root, [self.helper_visit(node.values), ])
        return root, [], []

    def visit_EnumeratorList(self, node: EnumeratorList, left):
        root = self.next_node("EnumeratorList")
        nodelist = []
        for i, child in enumerate(node.enumerators or []):
            nodelist.append(self.helper_visit(child))

        self.wrapper(root, nodelist)
        return root, [], []

    def visit_Enumerator(self, node: Enumerator, left):
        root = self.next_node("Enumerator")
        split = self.split_name(node.name)
        self.wrapper(root, [split, self.helper_visit(node.value)])
        return root, [], []
