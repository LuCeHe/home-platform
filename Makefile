all: clean build

build:
		python setup.py build_ext --inplace

install:
		python setup.py install

clean:
		rm -rf build 
		rm -f home_platform/pathfinding/*.so home_platform/pathfinding/*_wrap.* home_platform/pathfinding/astar.py