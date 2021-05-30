run:
	g++ main.cpp -std=c++17 `pkg-config --cflags --libs opencv`
	./a.out
