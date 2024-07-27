import sys

sys.path.append(f"{sys.path[0]}/../../..")

from register_rag.dashboard import Dashboard

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.launch()
