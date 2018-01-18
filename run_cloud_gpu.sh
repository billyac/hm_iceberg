BUCKET_NAME=cloud-test-kaggle
JOB_NAME=iceberg_$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
TRAIN_DATA=gs://$BUCKET_NAME/train.json
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $OUTPUT_PATH \
  --runtime-version 1.4  \
  --module-name trainer.train_cloud \
  --package-path trainer \
  --region us-central1 \
  --scale-tier BASIC_GPU \
  -- \
  --train-files $TRAIN_DATA \
  --fc-layers 512 512 \
  --dropouts 0.5 \
  --trainable-layers -1 # All layers are trainable
