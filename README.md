# hsys-dev-container

# 1 работа

# сборка
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# тесты
ctest --test-dir build --output-on-failure

# бенчмарки и графики
./build/work1/bench_add --benchmark_out=work1/results.json --benchmark_out_format=json

# создать venv если надо
py -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cd tools/benchmark-plot
python complexity_chart.py -j ../../work1/results.json
python speedup_chart.py -j ../../work1/results.json -r BM_AddCPU -t BM_AddGPU --baseline