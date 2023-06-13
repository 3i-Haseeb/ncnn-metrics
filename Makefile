run:
	rm -rf build
	cmake -S . -B build
	cmake --build ./build
	./build/main
	
