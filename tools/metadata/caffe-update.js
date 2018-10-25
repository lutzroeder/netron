/*jshint esversion: 6 */

const fs = require('fs');
const process = require('process');

var file = process.argv[2];

var data = fs.readFileSync(file, 'utf-8');

function update(search, replace) {
    var parts = data.split(search);
    if (parts.length != 2) {
        console.log('ERROR: Replace failed.');
        process.exit(1);
    }
    data = parts.join(replace);
}

function add(text) {
    data += text;
}

update(
`    CONVOLUTION = 4;`,
`    CONVOLUTION = 4;
    CROP = 40;`);

update(
`message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5]; // dropout ratio
}`,
`message DropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5]; // dropout ratio
  optional bool sample_weights_test = 2 [default = false]; // Sample weights for testing or use the mean
  optional bool scale_train = 3 [default = true];  // scale train or test phase
}`);

update(
`  optional WindowDataParameter window_data_param = 129;
}`,
`  optional string engine = 149 [default = ""];
  optional MultinodeLayerParameter multinode = 150;
  optional MnActivationParameter mn_activation_param = 151;
  optional BoxAnnotatorOHEMParameter box_annotator_ohem_param = 152;
  optional PSROIPoolingParameter psroi_pooling_param = 153;
  optional ROIPoolingParameter roi_pooling_param = 154; // 150 in yjxiong/caffe
  optional SpatialDropoutParameter spatial_dropout_param = 155;
  optional MnParamGradCompressParameter mn_grad_compress_param = 156;
  optional QuantizationParameter quantization_param = 158;
  optional ReorgParameter reorg_param = 159;
  optional BatchReductionParameter batch_reduction_param = 162;
  optional ShuffleChannelParameter shuffle_channel_param = 164;
  optional AnnotatedDataParameter annotated_data_param = 200;
  optional MultiBoxLossParameter multibox_loss_param = 201;
  optional PermuteParameter permute_param = 202;
  optional PriorBoxParameter prior_box_param = 203;
  optional DetectionOutputParameter detection_output_param = 204;
  optional DetectionEvaluateParameter detection_evaluate_param = 205;
  optional NormalizeParameter norm_param = 206;
  optional VideoDataParameter video_data_param = 207; // 140 in yjxiong/caffe
  optional SplitParameter split_param = 208;
  optional RegionLossParameter region_loss_param = 209;
  optional EvalDetectionParameter eval_detection_param = 301;
  optional ROIPoolingParameter roi_pooling_param_2 = 8266711;
  optional NormalizeBBoxParameter normalize_bbox_param = 8266712; // 149
  optional BNParameter bn_param = 1137; // 41 in ChenglongChen/batch_normalization, 137 in yjxiong/caffe
  optional bool force_backward = 8266713; // ???
}`);

update(
`  optional uint32 crop_size = 3 [default = 0];
  // mean_file and mean_value cannot be specified at the same time
  optional string mean_file = 4;
  // if specified can be repeated once (would subtract it from all the channels)
  // or can be repeated the same number of times as channels
  // (would subtract them from the corresponding channel)
  repeated float mean_value = 5;
  // Force the decoded image to have 3 color channels.
  optional bool force_color = 6 [default = false];
  // Force the decoded image to have 1 color channels.
  optional bool force_gray = 7 [default = false];
}`,
`  optional uint32 crop_size = 3 [default = 0];
  optional uint32 crop_h = 11 [default = 0];
  optional uint32 crop_w = 12 [default = 0];

  // mean_file and mean_value cannot be specified at the same time
  optional string mean_file = 4;
  // if specified can be repeated once (would substract it from all the channels)
  // or can be repeated the same number of times as channels
  // (would subtract them from the corresponding channel)
  repeated float mean_value = 5;
  // Force the decoded image to have 3 color channels.
  optional bool force_color = 6 [default = false];
  // Force the decoded image to have 1 color channels.
  optional bool force_gray = 7 [default = false];
  // Resize policy
  optional ResizeParameter resize_param = 8;
  // Noise policy
  optional NoiseParameter noise_param = 9;
  // Distortion policy
  optional DistortionParameter distort_param = 13;
  // Expand policy
  optional ExpansionParameter expand_param = 14;
  // Constraint for emitting the annotation after transformation.
  optional EmitConstraint emit_constraint = 10;
  // Resize the input randomly
  optional RandomResizeParameter random_resize_param = 15;
  optional RandomAspectRatioParameter random_aspect_ratio_param = 16;

    //will flip x flow if flow image input
  optional bool flow = 17 [default = false];

  optional bool bgr2rgb = 18 [ default = false ];
}

// Message that stores parameters used by data transformer for transformation
// policy
message NoiseParameter {
  //Probability of using this resize policy
  optional float prob = 1 [default = 0];
  // Histogram equalized
  optional bool hist_eq = 2 [default = false];
  // Color inversion
  optional bool inverse = 3 [default = false];
  // Grayscale
  optional bool decolorize = 4 [default = false];
  // Gaussian blur
  optional bool gauss_blur = 5 [default = false];

  // JPEG compression quality (-1 = no compression)
  optional float jpeg = 6 [default = -1];

  // Posterization
  optional bool posterize = 7 [default = false];

  // Erosion
  optional bool erode = 8 [default = false];

  // Salt-and-pepper noise
  optional bool saltpepper = 9 [default = false];

  optional SaltPepperParameter saltpepper_param = 10;

  // Local histogram equalization
  optional bool clahe = 11 [default = false];

  // Color space conversion
  optional bool convert_to_hsv = 12 [default = false];

  // Color space conversion
  optional bool convert_to_lab = 13 [default = false];
}

// Message that stores parameters used by data transformer for distortion policy
message DistortionParameter {
  // The probability of adjusting brightness.
  optional float brightness_prob = 1 [default = 0.0];
  // Amount to add to the pixel values within [-delta, delta].
  // The possible value is within [0, 255]. Recommend 32.
  optional float brightness_delta = 2 [default = 0.0];

  // The probability of adjusting contrast.
  optional float contrast_prob = 3 [default = 0.0];
  // Lower bound for random contrast factor. Recommend 0.5.
  optional float contrast_lower = 4 [default = 0.0];
  // Upper bound for random contrast factor. Recommend 1.5.
  optional float contrast_upper = 5 [default = 0.0];

  // The probability of adjusting hue.
  optional float hue_prob = 6 [default = 0.0];
  // Amount to add to the hue channel within [-delta, delta].
  // The possible value is within [0, 180]. Recommend 36.
  optional float hue_delta = 7 [default = 0.0];

  // The probability of adjusting saturation.
  optional float saturation_prob = 8 [default = 0.0];
  // Lower bound for the random saturation factor. Recommend 0.5.
  optional float saturation_lower = 9 [default = 0.0];
  // Upper bound for the random saturation factor. Recommend 1.5.
  optional float saturation_upper = 10 [default = 0.0];

  // The probability of randomly order the image channels.
  optional float random_order_prob = 11 [default = 0.0];
}

// Message that stores parameters used by data transformer for expansion policy
message ExpansionParameter {
  //Probability of using this expansion policy
  optional float prob = 1 [default = 1];

  // The ratio to expand the image.
  optional float max_expand_ratio = 2 [default = 1.];
}

// Condition for emitting annotations.
message EmitConstraint {
  enum EmitType {
    CENTER = 0;
    MIN_OVERLAP = 1;
  }
  optional EmitType emit_type = 1 [default = CENTER];
  // If emit_type is MIN_OVERLAP, provide the emit_overlap.
  optional float emit_overlap = 2;
}

message RandomResizeParameter {
  optional uint32 min_size = 1 [default = 0];
  optional uint32 max_size = 2 [default = 0];
  optional ResizeParameter resize_param = 3;
}

message RandomAspectRatioParameter {
  optional float min_area_ratio = 1 [default = 0.5];
  optional float max_area_ratio = 2 [default = 1];
  optional float aspect_ratio_change = 3 [default = 1];
  optional uint32 max_attempt = 4 [default = 10];
  optional ResizeParameter resize_param = 5;
}

message SaltPepperParameter {
  //Percentage of pixels
  optional float fraction = 1 [default = 0];
  repeated float value = 2;
}

message BNParameter {
  optional FillerParameter slope_filler = 1;
  optional FillerParameter bias_filler = 2;
  optional float momentum = 3 [default = 0.9];
  optional float eps = 4 [default = 1e-5];
  // If true, will use the moving average mean and std for training and test.
  // Will override the lr_param and freeze all the parameters.
  // Make sure to initialize the layer properly with pretrained parameters.
  optional bool frozen = 5 [default = false];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 6 [default = DEFAULT];
}

message BatchReductionParameter {
    repeated int32 level = 1;
    optional ReductionParameter reduction_param = 2;
    optional bool pos = 3 [default = false];
}
`);

add(
`
// Sample a bbox in the normalized space [0, 1] with provided constraints.
message Sampler {
  // Minimum scale of the sampled bbox.
  optional float min_scale = 1 [default = 1.];
  // Maximum scale of the sampled bbox.
  optional float max_scale = 2 [default = 1.];

  // Minimum aspect ratio of the sampled bbox.
  optional float min_aspect_ratio = 3 [default = 1.];
  // Maximum aspect ratio of the sampled bbox.
  optional float max_aspect_ratio = 4 [default = 1.];
}

// Constraints for selecting sampled bbox.
message SampleConstraint {
  // Minimum Jaccard overlap between sampled bbox and all bboxes in
  // AnnotationGroup.
  optional float min_jaccard_overlap = 1;
  // Maximum Jaccard overlap between sampled bbox and all bboxes in
  // AnnotationGroup.
  optional float max_jaccard_overlap = 2;

  // Minimum coverage of sampled bbox by all bboxes in AnnotationGroup.
  optional float min_sample_coverage = 3;
  // Maximum coverage of sampled bbox by all bboxes in AnnotationGroup.
  optional float max_sample_coverage = 4;

  // Minimum coverage of all bboxes in AnnotationGroup by sampled bbox.
  optional float min_object_coverage = 5;
  // Maximum coverage of all bboxes in AnnotationGroup by sampled bbox.
  optional float max_object_coverage = 6;
}

// Sample a batch of bboxes with provided constraints.
message BatchSampler {
  // Use original image as the source for sampling.
  optional bool use_original_image = 1 [default = true];

  // Constraints for sampling bbox.
  optional Sampler sampler = 2;

  // Constraints for determining if a sampled bbox is positive or negative.
  optional SampleConstraint sample_constraint = 3;

  // If provided, break when found certain number of samples satisfing the
  // sample_constraint.
  optional uint32 max_sample = 4;

  // Maximum number of trials for sampling to avoid infinite loop.
  optional uint32 max_trials = 5 [default = 100];
}

message NonMaximumSuppressionParameter {
  // Threshold to be used in nms.
  optional float nms_threshold = 1 [default = 0.3];
  // Maximum number of results to be kept.
  optional int32 top_k = 2;
  // Parameter for adaptive nms.
  optional float eta = 3 [default = 1.0];
}

message SaveOutputParameter {
  // Output directory. If not empty, we will save the results.
  optional string output_directory = 1;
  // Output name prefix.
  optional string output_name_prefix = 2;
  // Output format.
  //    VOC - PASCAL VOC output format.
  //    COCO - MS COCO output format.
  optional string output_format = 3;
  // If you want to output results, must also provide the following two files.
  // Otherwise, we will ignore saving results.
  // label map file.
  optional string label_map_file = 4;
  // A file which contains a list of names and sizes with same order
  // of the input DB. The file is in the following format:
  //    name height width
  //    ...
  optional string name_size_file = 5;
  // Number of test images. It can be less than the lines specified in
  // name_size_file. For example, when we only want to evaluate on part
  // of the test images.
  optional uint32 num_test_image = 6;
  // The resize parameter used in saving the data.
  optional ResizeParameter resize_param = 7;
}

// Message that stores parameters used by data transformer for resize policy
message ResizeParameter {
  //Probability of using this resize policy
  optional float prob = 1 [default = 1];

  enum Resize_mode {
    WARP = 1;
    FIT_SMALL_SIZE = 2;
    FIT_LARGE_SIZE_AND_PAD = 3;
  }
  optional Resize_mode resize_mode = 2 [default = WARP];
  optional uint32 height = 3 [default = 0];
  optional uint32 width = 4 [default = 0];
  // A parameter used to update bbox in FIT_SMALL_SIZE mode.
  optional uint32 height_scale = 8 [default = 0];
  optional uint32 width_scale = 9 [default = 0];

  enum Pad_mode {
    CONSTANT = 1;
    MIRRORED = 2;
    REPEAT_NEAREST = 3;
  }
  // Padding mode for BE_SMALL_SIZE_AND_PAD mode and object centering
  optional Pad_mode pad_mode = 5 [default = CONSTANT];
  // if specified can be repeated once (would fill all the channels)
  // or can be repeated the same number of times as channels
  // (would use it them to the corresponding channel)
  repeated float pad_value = 6;

  enum Interp_mode { //Same as in OpenCV
    LINEAR = 1;
    AREA = 2;
    NEAREST = 3;
    CUBIC = 4;
    LANCZOS4 = 5;
  }
  //interpolation for for resizing
  repeated Interp_mode interp_mode = 7;
}

// The normalized bounding box [0, 1] w.r.t. the input image size.
message NormalizedBBox {
  optional float xmin = 1;
  optional float ymin = 2;
  optional float xmax = 3;
  optional float ymax = 4;
  optional int32 label = 5;
  optional bool difficult = 6;
  optional float score = 7;
  optional float size = 8;
}

// Annotation for each object instance.
message Annotation {
  optional int32 instance_id = 1 [default = 0];
  optional NormalizedBBox bbox = 2;
}

// Group of annotations for a particular label.
message AnnotationGroup {
  optional int32 group_label = 1;
  repeated Annotation annotation = 2;
}

// An extension of Datum which contains "rich" annotations.
message AnnotatedDatum {
  enum AnnotationType {
    BBOX = 0;
  }
  optional Datum datum = 1;
  // If there are "rich" annotations, specify the type of annotation.
  // Currently it only supports bounding box.
  // If there are no "rich" annotations, use label in datum instead.
  optional AnnotationType type = 2;
  // Each group contains annotation for a particular class.
  repeated AnnotationGroup annotation_group = 3;
}

message MultinodeLayerParameter {
  // 0 means all nodes
  optional uint32 num_nodes = 1;
  // 0 or > num_nodes, means all nodes
  optional uint32 model_parts = 2 [default = 1];
}

message MnActivationParameter {
  // 0 means all nodes
  optional uint32 num_nodes_in = 1;
  // 0 means all nodes
  optional uint32 num_nodes_out = 2;
  // 0 or > num_nodes, means all nodes
  optional uint32 model_parts_in = 3 [default = 1];
  // 0 or > num_nodes, means all nodes
  optional uint32 model_parts_out = 4 [default = 1];
  optional bool need_reduce = 5 [default = true];
}

message BoxAnnotatorOHEMParameter {
  required uint32 roi_per_img = 1; // number of rois for training
  optional int32 ignore_label = 2 [default = -1]; // ignore_label in scoring
}

message PSROIPoolingParameter {
   required float spatial_scale = 1; 
   required int32 output_dim = 2; // output channel number
   required int32 group_size = 3; // number of groups to encode position-sensitive score maps
}

// Message that stores parameters used by ROIPoolingLayer
message ROIPoolingParameter {
  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in height and width or as Y, X pairs.
  optional uint32 pooled_h = 1 [default = 0]; // The pooled output height
  optional uint32 pooled_w = 2 [default = 0]; // The pooled output width
  // Multiplicative spatial scale factor to translate ROI coords from their
  // input scale to the scale used when pooling
  optional float spatial_scale = 3 [default = 1];
  repeated uint32 pooled_size = 4; // pooled size for 3D blobs
}

message SpatialDropoutParameter {
  optional float dropout_ratio = 1 [default = 0.5]; // dropout ratio
}

message MnParamGradCompressParameter {
  repeated bool param_grad_compress_enable = 1;
}

// Message for layers with reduced word with arithmetic
message QuantizationParameter{
  enum Precision {
    DYNAMIC_FIXED_POINT = 0;
  }
  optional Precision precision = 1 [default = DYNAMIC_FIXED_POINT];
  enum Rounding {
    NEAREST = 0;
  }
  optional Rounding rounding_scheme = 2 [default = NEAREST];
  // Dynamic fixed point word width
  optional uint32 bw_layer_in = 3 [default = 32];
  optional uint32 bw_layer_out = 4 [default = 32];
  optional uint32 bw_params = 5 [default = 32];
  repeated int32 fl_layer_in = 6;
  repeated int32 fl_layer_out = 7;
  repeated int32 fl_params = 8;
  repeated float scale_in = 20;
  repeated float scale_out = 21;
  repeated float scale_params = 22;
}

message ReorgParameter {
  optional uint32 stride = 1;
  optional bool reverse = 2 [default = false];
}

message AnnotatedDataParameter {
  // Define the sampler.
  repeated BatchSampler batch_sampler = 1;
  // Store label name and label id in LabelMap format.
  optional string label_map_file = 2;
  // If provided, it will replace the AnnotationType stored in each
  // AnnotatedDatum.
  optional AnnotatedDatum.AnnotationType anno_type = 3;
}

// Message that store parameters used by MultiBoxLossLayer
message MultiBoxLossParameter {
  // Localization loss type.
  enum LocLossType {
    L2 = 0;
    SMOOTH_L1 = 1;
  }
  optional LocLossType loc_loss_type = 1 [default = SMOOTH_L1];
  // Confidence loss type.
  enum ConfLossType {
    SOFTMAX = 0;
    LOGISTIC = 1;
  }
  optional ConfLossType conf_loss_type = 2 [default = SOFTMAX];
  // Weight for localization loss.
  optional float loc_weight = 3 [default = 1.0];
  // Number of classes to be predicted. Required!
  optional uint32 num_classes = 4;
  // If true, bounding box are shared among different classes.
  optional bool share_location = 5 [default = true];
  // Matching method during training.
  enum MatchType {
    BIPARTITE = 0;
    PER_PREDICTION = 1;
  }
  optional MatchType match_type = 6 [default = PER_PREDICTION];
  // If match_type is PER_PREDICTION, use overlap_threshold to
  // determine the extra matching bboxes.
  optional float overlap_threshold = 7 [default = 0.5];
  // Use prior for matching.
  optional bool use_prior_for_matching = 8 [default = true];
  // Background label id.
  optional uint32 background_label_id = 9 [default = 0];
  // If true, also consider difficult ground truth.
  optional bool use_difficult_gt = 10 [default = true];
  // If true, perform negative mining.
  // DEPRECATED: use mining_type instead.
  optional bool do_neg_mining = 11;
  // The negative/positive ratio.
  optional float neg_pos_ratio = 12 [default = 3.0];
  // The negative overlap upperbound for the unmatched predictions.
  optional float neg_overlap = 13 [default = 0.5];
  // Type of coding method for bbox.
  optional PriorBoxParameter.CodeType code_type = 14 [default = CORNER];
  // If true, encode the variance of prior box in the loc loss target instead of
  // in bbox.
  optional bool encode_variance_in_target = 16 [default = false];
  // If true, map all object classes to agnostic class. It is useful for learning
  // objectness detector.
  optional bool map_object_to_agnostic = 17 [default = false];
  // If true, ignore cross boundary bbox during matching.
  // Cross boundary bbox is a bbox who is outside of the image region.
  optional bool ignore_cross_boundary_bbox = 18 [default = false];
  // If true, only backpropagate on corners which are inside of the image
  // region when encode_type is CORNER or CORNER_SIZE.
  optional bool bp_inside = 19 [default = false];
  // Mining type during training.
  //   NONE : use all negatives.
  //   MAX_NEGATIVE : select negatives based on the score.
  //   HARD_EXAMPLE : select hard examples based on "Training Region-based Object Detectors with Online Hard Example Mining", Shrivastava et.al.
  enum MiningType {
    NONE = 0;
    MAX_NEGATIVE = 1;
    HARD_EXAMPLE = 2;
  }
  optional MiningType mining_type = 20 [default = MAX_NEGATIVE];
  // Parameters used for non maximum suppression durig hard example mining.
  optional NonMaximumSuppressionParameter nms_param = 21;
  optional int32 sample_size = 22 [default = 64];
  optional bool use_prior_for_nms = 23 [default = false];
}

message PermuteParameter {
  // The new orders of the axes of data. Notice it should be with
  // in the same range as the input data, and it starts from 0.
  // Do not provide repeated order.
  repeated uint32 order = 1;
}

// Message that stores parameters used by NormalizeBBoxLayer
message NormalizeBBoxParameter {
  optional bool across_spatial = 1 [default = true];
  // Initial value of scale. Default is 1.0 for all
  optional FillerParameter scale_filler = 2;
  // Whether or not scale parameters are shared across channels.
  optional bool channel_shared = 3 [default = true];
  // Epsilon for not dividing by zero while normalizing variance
  optional float eps = 4 [default = 1e-10];
}

message PriorBoxParameter {
  // Encode/decode type.
  enum CodeType {
    CORNER = 1;
    CENTER_SIZE = 2;
    CORNER_SIZE = 3;
  }
  // Minimum box size (in pixels). Required!
  repeated float min_size = 1;
  // Maximum box size (in pixels). Required!
  repeated float max_size = 2;
  // Various of aspect ratios. Duplicate ratios will be ignored.
  // If none is provided, we use default ratio 1.
  repeated float aspect_ratio = 3;
  // If true, will flip each aspect ratio.
  // For example, if there is aspect ratio "r",
  // we will generate aspect ratio "1.0/r" as well.
  optional bool flip = 4 [default = true];
  // If true, will clip the prior so that it is within [0, 1]
  optional bool clip = 5 [default = false];
  // Variance for adjusting the prior bboxes.
  repeated float variance = 6;
  // By default, we calculate img_height, img_width, step_x, step_y based on
  // bottom[0] (feat) and bottom[1] (img). Unless these values are explicitely
  // provided.
  // Explicitly provide the img_size.
  optional uint32 img_size = 7;
  // Either img_size or img_h/img_w should be specified; not both.
  optional uint32 img_h = 8;
  optional uint32 img_w = 9;

  // Explicitly provide the step size.
  optional float step = 10;
  // Either step or step_h/step_w should be specified; not both.
  optional float step_h = 11;
  optional float step_w = 12;

  // Offset to the top left corner of each cell.
  optional float offset = 13 [default = 0.5];
}

// Message that store parameters used by DetectionOutputLayer
message DetectionOutputParameter {
  // Number of classes to be predicted. Required!
  optional uint32 num_classes = 1;
  // If true, bounding box are shared among different classes.
  optional bool share_location = 2 [default = true];
  // Background label id. If there is no background class,
  // set it as -1.
  optional int32 background_label_id = 3 [default = 0];
  // Parameters used for non maximum suppression.
  optional NonMaximumSuppressionParameter nms_param = 4;
  // Parameters used for saving detection results.
  optional SaveOutputParameter save_output_param = 5;
  // Type of coding method for bbox.
  optional PriorBoxParameter.CodeType code_type = 6 [default = CORNER];
  // If true, variance is encoded in target; otherwise we need to adjust the
  // predicted offset accordingly.
  optional bool variance_encoded_in_target = 8 [default = false];
  // Number of total bboxes to be kept per image after nms step.
  // -1 means keeping all bboxes after nms step.
  optional int32 keep_top_k = 7 [default = -1];
  // Only consider detections whose confidences are larger than a threshold.
  // If not provided, consider all boxes.
  optional float confidence_threshold = 9;
  // If true, visualize the detection results.
  optional bool visualize = 10 [default = false];
  // The threshold used to visualize the detection results.
  optional float visualize_threshold = 11;
  // If provided, save outputs to video file.
  optional string save_file = 12;
  // Parameters used for non maximum suppression.
  // Threshold to be used in nms.
  optional float nms_threshold = 32768 [default = 0.3];
  // Maximum number of results to be kept.
  optional int32 top_k = 32769;
}

// Message that store parameters used by DetectionEvaluateLayer
message DetectionEvaluateParameter {
  // Number of classes that are actually predicted. Required!
  optional uint32 num_classes = 1;
  // Label id for background class. Needed for sanity check so that
  // background class is neither in the ground truth nor the detections.
  optional uint32 background_label_id = 2 [default = 0];
  // Threshold for deciding true/false positive.
  optional float overlap_threshold = 3 [default = 0.5];
  // If true, also consider difficult ground truth for evaluation.
  optional bool evaluate_difficult_gt = 4 [default = true];
  // A file which contains a list of names and sizes with same order
  // of the input DB. The file is in the following format:
  //    name height width
  //    ...
  // If provided, we will scale the prediction and ground truth NormalizedBBox
  // for evaluation.
  optional string name_size_file = 5;
  // The resize parameter used in converting NormalizedBBox to original image.
  optional ResizeParameter resize_param = 6;
}

// Message that stores parameters used by NormalizeLayer
message NormalizeParameter {
  optional bool across_spatial = 1 [default = true];
  // Initial value of scale. Default is 1.0 for all
  optional FillerParameter scale_filler = 2;
  // Whether or not scale parameters are shared across channels.
  optional bool channel_shared = 3 [default = true];
  // Epsilon for not dividing by zero while normalizing variance
  optional float eps = 4 [default = 1e-10];
}

message VideoDataParameter{
  enum VideoType {
    WEBCAM = 0;
    VIDEO = 1;
  }
  optional VideoType video_type = 1 [default = WEBCAM];
  optional int32 device_id = 2 [default = 0];
  optional string video_file = 3;
  // Number of frames to be skipped before processing a frame.
  optional uint32 skip_frames = 4 [default = 0];
}

message SplitParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    MKL2017 = 3;
    MKLDNN = 4;
  }
  optional Engine engine = 1 [default = DEFAULT];
}

message RegionLossParameter{
  //Yolo 9000
  optional uint32 side = 1 [default = 13];
  optional uint32 num_class = 2 [default = 20];
  optional uint32 bias_match = 3 [default = 1];
  optional uint32 coords = 4 [default = 4];
  optional uint32 num = 5 [default = 5];
  optional uint32 softmax = 6 [default = 1];
  optional float jitter = 7 [default = 0.2];
  optional uint32 rescore = 8 [default = 1];

  optional float object_scale = 9 [default = 1.0];
  optional float class_scale = 10 [default = 1.0];
  optional float noobject_scale = 11 [default = 0.5];
  optional float coord_scale = 12 [default = 5.0];
  optional uint32 absolute = 13 [default = 1];
  optional float thresh = 14 [default = 0.2];
  optional uint32 random = 15 [default = 1];
  repeated float biases = 16;
  optional string softmax_tree = 17;
  optional string class_map = 18;
}

message EvalDetectionParameter {
  enum ScoreType {
    OBJ = 0;
    PROB = 1;
    MULTIPLY = 2;
  }
  // Yolo detection evaluation layer
  optional uint32 side = 1 [default = 7];
  optional uint32 num_class = 2 [default = 20];
  optional uint32 num_object = 3 [default = 2];
  optional float threshold = 4 [default = 0.5];
  optional bool sqrt = 5 [default = true];
  optional bool constriant = 6 [default = true];
  optional ScoreType score_type = 7 [default = MULTIPLY];
  optional float nms = 8 [default = -1];
  repeated float biases = 9;
}

message ShuffleChannelParameter {
  optional uint32 group = 1 [default = 1]; // The number of group
}

`);

fs.writeFileSync(file, data, 'utf-8');
