for SEED in 0 1 2 3 4 5
do
    for SIZE in 32 64 128
        do
            for WEIGHT in 5 10
                do
                    python run_gradient.py --name=heart --variable=age --lr=0.1 --size=$SIZE --iter=50 --dist-weight=$WEIGHT --seed=$SEED
                done
        done
done
