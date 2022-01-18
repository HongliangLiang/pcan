from pycparser import parse_file

from c_extractor.extractor import GeneralVisitor

if __name__ == "__main__":
    ast = parse_file('../test/error1.cpp', use_cpp=True)

    v = GeneralVisitor()
    v.visit(ast)

    print(v.build_graph())
