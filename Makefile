all: cbin/vqtree cbin/ann

cbin:
	mkdir cbin

cbin/vqtree: cbin vqtree_main.cpp vqtree.cpp onlineaverage.cpp
	clang++ -std=c++11 -O3 vqtree_main.cpp -o cbin/vqtree -g

cbin/ann: cbin ann_main.cpp vqtree.cpp onlineaverage.cpp
	clang++ -std=c++11 -Wall -pedantic -O3 ann_main.cpp -o cbin/ann -g
