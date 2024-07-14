import sys

sys.path.append(f"{sys.path[0]}/..")

from loader import SqliteLoader

loader = SqliteLoader(
    db_path="/home/yumuzhihan/Documents/xwechat_files/wxid_v0o28t80xtgs22_a67f/msg/file/2024-06/病毒日志/196-中心196/kvirusinfo.dat"
)

documents = loader.load_file()

for document in documents:
    print(document.page_content)
