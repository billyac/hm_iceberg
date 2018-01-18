gcloud ml-engine local train \
	--module-name trainer.train_cloud \
	--package-path trainer \
	--job-dir output \
	-- \
	--train-files data/train.json \
        --fc-layers 256 \
	--dropouts 0.5 \
	--trainable-layers 6
