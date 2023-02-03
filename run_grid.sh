for SEED in 6 7 8 9
do
    for SIZE in 32 64 128
        do
            for WEIGHT in 0 0.001 0.01 0.1 1 10
                do
                    python run_gradient.py --name=xor --variable=x1 --lr=0.1 --size=$SIZE --iter=50 --dist-weight=$WEIGHT --seed=$SEED
                done
        done
done
