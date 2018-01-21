BUCKET_NAME=cloud-test-kaggle
JOB_NAME=iceberg_$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
TRAIN_DATA=gs://$BUCKET_NAME/train.json
TEST_DATA=gs://$BUCKET_NAME/test.json
gcloud ml-engine jobs submit training $JOB_NAME \
  --job-dir $OUTPUT_PATH \
  --runtime-version 1.4  \
  --module-name trainer.train_cloud \
  --package-path trainer \
  --region us-central1 \
  --scale-tier BASIC_GPU \
  -- \
  --train-files $TRAIN_DATA \
  --decay 0.01 \
  --patience 50  \
  --rotation-range 20  \
  --horizontal-flip True  \
  --vertical-flip True  \
  --width-shift-range 0.1  \
  --height-shift-range 0.1  \
  --zoom-range 0.1  \
  --model-name InceptionResNet2  \
  --fc-layers 512 512 256  \
  --dropouts  0.35 \
  --do-predict-test True  \
  --test-file $TEST_DATA

  # Unsed parameters
  # Trainning
  # --learning-rate 0.001 \
  # --train-batch-size 64 \
  # --steps-per-epoch 15 \
  # --num-epochs 100  \
  # Cross validation
  # --cv 1  \
  # --val-ratio 0.2  \
  # Predict test data

  # Transfer learning model
  # --pooling 'max'  \
  # --trainable-layers -1  \
