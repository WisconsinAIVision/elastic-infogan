for i in {1..50}
do 
	CUDA_VISIBLE_DEVICES=0 python main.py --ind $i --mytemp 1 --klwt 10 
done

