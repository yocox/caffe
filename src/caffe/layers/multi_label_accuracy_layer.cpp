// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;


namespace caffe {

template <typename Dtype>
void MultiLabelAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number of instances";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels())
    << "The data and label should have the same number of channels";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data and label should have the same height";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data and label should have the same width";
  // Top will contain:
  // top[0] = Sensitivity or Recall (TP/P),
  // top[1] = Specificity (TN/N),
  // top[2] = accuracy ((TP+TN) / (P+N))
  // top[3] = Precision (TP / (TP + FP))
  // top[4] = F1 Score (2 TP / (2 TP + FP + FN))
  // top[5] = overall accuracy (all-label-true / num)
  (*top)[0]->Reshape(1, 6, 1, 1);
}

template <typename Dtype>
Dtype MultiLabelAccuracyLayer<Dtype>::Forward_cpu(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  Dtype true_positive = 0;
  Dtype false_positive = 0;
  Dtype true_negative = 0;
  Dtype false_negative = 0;
  int count_pos = 0;
  int count_neg = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  // Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int count = bottom[0]->count();
  // overall accuracy
  Dtype overall_true = 0;
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  for (int i=0; i<num; ++i){
	int flag = 1;
	for (int j=0; j<channels; ++j){
		int label = static_cast<int>(bottom_label[i*channels+j]);
		if(label > 0){
			flag *= (bottom_data[i*channels+j] >= 0);
		}
		else{
			flag *= (bottom_data[i*channels+j] < 0);
		}
	}
	overall_true += flag;
  }

  // other values
  for (int ind = 0; ind < count; ++ind) {
    // Accuracy
    int label = static_cast<int>(bottom_label[ind]);
    if (label > 0) {
    // Update Positive accuracy and count
      true_positive += (bottom_data[ind] >= 0);
      false_negative += (bottom_data[ind] < 0);
      count_pos++;
    }
    if (label < 0) {
    // Update Negative accuracy and count
      true_negative += (bottom_data[ind] < 0);
      false_positive += (bottom_data[ind] >= 0);
      count_neg++;
    }
  }
  Dtype sensitivity = (count_pos > 0)? (true_positive / count_pos) : 0;
  Dtype specificity = (count_neg > 0)? (true_negative / count_neg) : 0;
  Dtype accuracy = (count_pos+count_neg > 0)? ((true_positive+true_negative) / (count_pos+count_neg)) : 0;
  Dtype precission = (true_positive > 0)?
    (true_positive / (true_positive + false_positive)) : 0;
  Dtype f1_score = (true_positive > 0)?
    2 * true_positive /
    (2 * true_positive + false_positive + false_negative) : 0;
  Dtype overall_accuracy = (num > 0)? (overall_true / num) : 0;

  DLOG(INFO) << "Sensitivity: " << sensitivity;
  DLOG(INFO) << "Specificity: " << specificity;
  DLOG(INFO) << "Accuracy: " << accuracy;
  DLOG(INFO) << "Precission: " << precission;
  DLOG(INFO) << "F1 Score: " << f1_score;
  DLOG(INFO) << "Overall accuracy: " << overall_accuracy;
  (*top)[0]->mutable_cpu_data()[0] = sensitivity;
  (*top)[0]->mutable_cpu_data()[1] = specificity;
  (*top)[0]->mutable_cpu_data()[2] = accuracy;
  (*top)[0]->mutable_cpu_data()[3] = precission;
  (*top)[0]->mutable_cpu_data()[4] = f1_score;
  (*top)[0]->mutable_cpu_data()[5] = overall_accuracy;

  // MultiLabelAccuracy should not be used as a loss function.
  return Dtype(0);
}

INSTANTIATE_CLASS(MultiLabelAccuracyLayer);

}  // namespace caffe
