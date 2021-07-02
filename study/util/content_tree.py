from pathlib import Path

tree_str = ""


def generate_tree(pathname, n=0):
    """
    :param pathname: 面向对象的路径
    :param n: 控制路径的空格输出次数
    :return:
    """
    global tree_str
    if pathname.is_file():
        if pathname.name != "__init__.py":
            tree_str += '    |' * n + '-' + pathname.name + '\n'
    if pathname.is_dir():
        if pathname.name != ".idea":
            tree_str += '    |' * n + '-' + pathname.name + "\\" + '\n'
            for cp in pathname.iterdir():
                generate_tree(cp, n + 1)


# 输出目录树
if __name__ == "__main__":
    path = '../../study'
    generate_tree(Path(path), 0)
    print(tree_str)
