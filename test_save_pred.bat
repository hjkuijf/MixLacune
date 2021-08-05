call ./build.bat

docker run --rm --runtime nvidia --memory=10g --gpus="device=0"^
       -v %~dp0\test\:/input/ -v %~dp0\output\:/output/^
       mixlacune

docker run --rm^
 -v %~dp0\output\:/output/^
 -v %~dp0\test\:/input/^
 python:3.7-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

if %ERRORLEVEL% == 0 (
	echo "Tests successfully passed..."
)
else
(
	echo "Expected output was not found..."
)