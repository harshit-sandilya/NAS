pip install -r requirements.txt

for i in {51..100}
do
    echo "Running iteration $i..."
    { time python train-rl.py > output$i.log 2>&1; } 2>> time.log
done
