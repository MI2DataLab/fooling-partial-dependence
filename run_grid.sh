for SEED in 0 1 2 3 4 5
do
    for SIZE in 4 8 16 32 64 128 256 512
        do
            python run_gradient.py --name=heart --variable=age --lr=0.1 --size=$SIZE --iter=50 --dist-weight=0.01 --seed=$SEED
        done
done