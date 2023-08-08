export CUDA_VISIBLE_DEVICES=0
NUM_PROC=1
FILE=odir_train_no_fold.py
PORT=29714
EPOCH=10
BATCH_SIZE=64
OUTPUT=./final_for_paper_table_mix_fusion_cheer_up/
IMAGE_SIZE=384
MODEL=all
debug:
	export CUDA_VISIBLE_DEVICES=2
	python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m torch.distributed.launch --master_port=$(PORT) --nproc_per_node=1 $(FILE) \
	--batch-size $(BATCH_SIZE) --output $(OUTPUT) --std 0.254417 0.17148255 0.0995115 --mean 0.4407552 0.28228086 0.15446076 --img-size $(IMAGE_SIZE)


train_dist:
	python -m torch.distributed.launch --master_port=$(PORT) --nproc_per_node=$(NUM_PROC) \
	$(FILE) --epochs $(EPOCH) --batch-size $(BATCH_SIZE) --workers 16 \
	--output $(OUTPUT) --img-size $(IMAGE_SIZE) --model $(MODEL) \
	--std 0.254417 0.17148255 0.0995115 --mean 0.4407552 0.28228086 0.15446076
train_dist_chexpert:
	python -m torch.distributed.launch --master_port=$(PORT) --nproc_per_node=$(NUM_PROC) \
	$(FILE) --epochs $(EPOCH) --batch-size $(BATCH_SIZE) --workers 16 \
	--output $(OUTPUT) --img-size $(IMAGE_SIZE) --model $(MODEL) \
	--std 0.28884754 --mean 0.50288445
eval:
	python eval_csv.py
clean:
	rm -rf $(OUTPUT)
get_score:
	@echo ""
	@python eval_csv.py
	@python handle_submit_file.py
	@python evaluation.py

get_score_mix:
	@echo ""
	# @python eval_csv_mix.py
	@python handle_submit_file.py
	@python evaluation.py
.PHONY: train_dist,debug,clean,eval,get_score