"""Tiny helpers so notebooks are written as Python lists of cells and built reproducibly."""
import nbformat as nbf


def md(src):
    return nbf.v4.new_markdown_cell(src.strip("\n"))


def code(src):
    return nbf.v4.new_code_cell(src.strip("\n"))
