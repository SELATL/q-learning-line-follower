"""실행: python main.py / python main.py --manual"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="2D line follower simulator")
    parser.add_argument('--manual', action='store_true', help='수동 조작으로 시작')
    parser.add_argument('--snapshot', help='화면 대신 PNG 파일 저장')
    args = parser.parse_args()
    if args.snapshot:
        import matplotlib
        matplotlib.use('Agg')
    from line_follower.visualization import Viewer
    viewer = Viewer(manual=args.manual)
    if args.snapshot:
        for i in range(200):
            viewer.update(i)
        viewer.fig.savefig(args.snapshot, dpi=150)
    else:
        viewer.show()

if __name__ == '__main__':
    main()
