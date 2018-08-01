all: clean build

build:
		python3 setup.py build_ext

install:
		python3 setup.py install

clean:
		rm -rf build 
		rm -f home_platform/pathfinding/*.so home_platform/pathfinding/*_wrap.* home_platform/pathfinding/astar.py