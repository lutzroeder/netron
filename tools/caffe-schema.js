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
  optional SqueezeConvolutionParameter squeeze_convolution_param = 187; // synopsys-caffe
  optional SqueezeInnerProductParameter squeeze_inner_product_param = 188; // synopsys-caffe
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
  optional YoloDetectionOutputParameter yolo_detection_output_param = 601; // eric612/MobileNet-YOLO
  optional Yolov3DetectionOutputParameter yolov3_detection_output_param = 602; // eric612/MobileNet-YOLO
  optional Yolov3Parameter yolov3_param = 603; // 603 in eric612/Caffe-YOLOv3-Windows
  optional ROIPoolingParameter roi_pooling_param_2 = 8266711;
  optional NormalizeBBoxParameter normalize_bbox_param = 8266712; // 149
  optional BNParameter bn_param = 1137; // 41 in ChenglongChen/batch_normalization, 137 in yjxiong/caffe
  optional InterpParameter interp_param = 2143; // 143 indeeplab-public-ver2
  optional CoeffScheduleParameter coeff_schedule_param = 3148; // 148 synopsys-caffe
  optional AugmentationParameter augmentation_param = 3149; // // 149 synopsys-caffe
  optional CorrelationParameter correlation_param = 3150; // 150 synopsys-caffe
  optional L1LossParameter l1_loss_param = 3151; // 2151 synopsys-caffe
  optional WriterParameter writer_param = 3152; // 152 synopsys-caffe
  optional ReaderParameter reader_param = 3153; // 153 synopsys-caffe
  optional MeanParameter mean_param = 3154; // 154 synopsys-caffe
  optional ResampleParameter resample_param = 3155; // 155
  optional DownsampleParameter downsample_param = 3156; // 156
  optional LpqLossParameter lpq_loss_param = 3158; // 158
  optional FlowWarpParameter flow_warp_param = 3159; // 159
  optional AccumParameter accum_param = 3160; // 160
  optional BlackAugmentationParameter black_augmentation_param = 3161; // 161
  optional bool reshape_every_iter = 3157 [default = true]; // 162
  optional YoloV2LossParameter yolo_v2_loss_param = 3198; // 198
  optional YoloV3LossParameter yolo_v3_loss_param = 3199; // 199
  optional BoxOutputParameter box_output_param = 4151; // 151 in gdlg/panoramic-object-detection
  optional RingPadParameter ring_pad_param = 4158; // 158 in gdlg/panoramic-object-detection
  optional bool force_backward = 4000; // ???
  optional SmoothL1LossParameter smooth_l1_loss_param = 5148; // 148 in mahyarnajibi/caffe-ssh
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

// Message that stores parameters used by InterpLayer
message InterpParameter {
  optional int32 height = 1 [default = 0]; // Height of output
  optional int32 width = 2 [default = 0]; // Width of output
  optional int32 zoom_factor = 3 [default = 1]; // zoom factor
  optional int32 shrink_factor = 4 [default = 1]; // shrink factor
  optional int32 pad_beg = 5 [default = 0]; // padding at begin of input
  optional int32 pad_end = 6 [default = 0]; // padding at end of input
}
`);

update(
`  // kernel_h = bottom->height and kernel_w = bottom->width
  optional bool global_pooling = 12 [default = false];
`,
`  // kernel_h = bottom->height and kernel_w = bottom->width
  optional bool global_pooling = 12 [default = false];
  // Specify floor/ceil mode
  optional bool ceil_mode = 1013 [default = true]; // 13 in https://github.com/BVLC/caffe/pull/3057
`);

// foss-for-synopsys-dwc-arc-processors/synopsys-caffe
update(
`  // Prefetch queue (Increase if data feeding bandwidth varies, within the
  // limit of device memory for GPU training)
  optional uint32 prefetch = 10 [default = 4];
`,
`  // Prefetch queue (Increase if data feeding bandwidth varies, within the
  // limit of device memory for GPU training)
  optional uint32 prefetch = 10 [default = 4];
  //To store last layer feature map size for yolo
  repeated uint32 side = 11;
  // Read data from BinaryDB files using multiple threads. If this parameter
  // is set to ZERO, each top blob will get a separate thread.
  optional uint32 disk_reader_threads = 4001 [default = 1];
  // If set to true, BinaryDB uses a pretty fast method for initializing the source data.
  // Without this BinaryDB practically fails to load huge datasets.
  optional bool huge_video_dataset = 7000 [default = false];
  // Use a non-negative value, to load only this number of dataset samples in binarydb_*
  optional int32 limit_samples = 7001 [default = -1];
  optional string preselection_file = 3001; // Load a text file which specifies a label for each data sample. Use preselection_label to filter the data for one specific label.
  optional int32 preselection_label = 3002;
  optional int32 range_start = 3003 [default = 0]; //0 = start with first
  optional int32 range_end = 3004 [default = -1]; //-1 = go until end
  optional bool rand_permute = 3005 [default = false];
  optional RANDPERMORDER rand_permute_order = 3006 [default = FIRST_PERMUTE_THEN_RANGE]; // permute indices first, then extract range (or other way around)
  optional uint32 rand_permute_seed = 3007 [default = 0]; // In BinaryDB (and webp), 0 means using timer randomization!
  repeated uint32 slice_point = 3008;
  repeated CHANNELENCODING encoding = 3009;
  optional bool verbose = 3010 [default = false];
  repeated float subtract = 3011;
  optional uint32 permute_every_iter = 3012 [default = 0];
  optional uint32 block_size = 3013 [default = 0];
  enum RANDPERMORDER {
    FIRST_PERMUTE_THEN_RANGE = 0;
    FIRST_RANGE_THEN_PERMUTE = 1;
  }
  enum CHANNELENCODING {
    UINT8 = 1;
    UINT16FLOW = 2;
    BOOL1 = 3;
  }
`)

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
   optional float pad_ratio = 6 [default = 0]; // 4 in gdlg/panoramic-object-detection
   optional bool ringpad = 5 [default = false]; // gdlg/panoramic-object-detection
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
  optional int32 yolo_data_type = 4 [default = 0]; // eric612/Caffe-YOLOv3-Windows
  optional float yolo_data_jitter = 5 [default = 0.3]; // eric612/Caffe-YOLOv3-Windows
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

message CoeffScheduleParameter {
  optional float half_life = 1 [default = 1];
  optional float initial_coeff = 2 [default = 1];
  optional float final_coeff = 3 [default = 1];
}

// Message describing distribution of augmentation parameters
message AugmentationParameter {
  optional uint32 crop_width = 33 [default = 0];
  optional uint32 crop_height = 34 [default = 0];
  optional string write_augmented = 2 [default = ""];
  optional float max_multiplier = 3 [default = 255.];
  optional bool augment_during_test = 4 [default = false];
  optional uint32 recompute_mean = 5 [default = 0]; // number of iterations to recompute mean (0 - do not recompute)
  optional string write_mean = 6 [default = ""];
  optional bool mean_per_pixel = 7 [default = true]; // if the mean is computed for each pixel or for the whole channel
  repeated float mean = 18; // Eddy: Per pixel RGB mean to subtract
  optional string mode = 8 [default = "add"]; // can be "add" or "replace" or "regenerate"
  optional uint32 bottomwidth = 80 [default = 0];
  optional uint32 bottomheight = 81 [default = 0];
  optional uint32 num = 82 [default = 0];

  repeated float chromatic_eigvec = 83;

  // Spatial
  optional RandomGeneratorParameter mirror = 10;
  optional RandomGeneratorParameter translate = 11 ;
  optional RandomGeneratorParameter rotate = 12 ;
  optional RandomGeneratorParameter zoom = 13 ;
  optional RandomGeneratorParameter squeeze = 14 ;
  optional RandomGeneratorParameter translate_x = 15 ;
  optional RandomGeneratorParameter translate_y = 16 ;


  // Chromatic
  optional RandomGeneratorParameter gamma = 35 ;
  optional RandomGeneratorParameter brightness = 36 ;
  optional RandomGeneratorParameter contrast = 37 ;
  optional RandomGeneratorParameter color = 38 ;

  // Chromatic-Eigen
  optional RandomGeneratorParameter lmult_pow = 20 ;
  optional RandomGeneratorParameter lmult_mult = 21 ;
  optional RandomGeneratorParameter lmult_add = 22 ;
  optional RandomGeneratorParameter sat_pow = 23 ;
  optional RandomGeneratorParameter sat_mult = 24 ;
  optional RandomGeneratorParameter sat_add = 25 ;
  optional RandomGeneratorParameter col_pow = 26 ;
  optional RandomGeneratorParameter col_mult = 27 ;
  optional RandomGeneratorParameter col_add = 28 ;
  optional RandomGeneratorParameter ladd_pow = 29 ;
  optional RandomGeneratorParameter ladd_mult = 30 ;
  optional RandomGeneratorParameter ladd_add = 31 ;
  optional RandomGeneratorParameter col_rotate = 32 ;

  // Effect
  optional RandomGeneratorParameter fog_amount = 100 ;
  optional RandomGeneratorParameter fog_size = 101 ;
  optional RandomGeneratorParameter motion_blur_angle = 102 ;
  optional RandomGeneratorParameter motion_blur_size = 103 ;
  optional RandomGeneratorParameter shadow_angle = 104 ;
  optional RandomGeneratorParameter shadow_distance = 105 ;
  optional RandomGeneratorParameter shadow_strength = 106 ;
  optional RandomGeneratorParameter noise = 107 ;
}

message BlackAugmentationParameter {
    optional RandomGeneratorParameter black = 10;
    optional RandomGeneratorParameter border = 11;
}

// Message that stores parameters used by CorrelationLayer
message CorrelationParameter {
  optional uint32 pad = 2 [default = 0]; // The padding size (equal in Y, X)
  optional uint32 kernel_size = 3; // The kernel size (square)
  optional uint32 max_displacement = 4; // The maximum displacement (square)
  optional uint32 stride_1 = 5 [default = 1]; // The stride in blob 1 (equal in Y, X)
  optional uint32 stride_2 = 6 [default = 1]; // The stride in blob 2 (equal in Y, X)

  // For Correlation1D:
  optional int32 single_direction = 8 [default = 0]; // Correlate only to the left (-1) or right (1)

  optional bool do_abs = 7 [default = false]; // Use absolute value of result
  enum CorrelationType {
    MULTIPLY = 0;
    SUBTRACT = 1;
  }
  optional CorrelationType correlation_type = 15 [default = MULTIPLY]; // Multiplicative is normal correlation
}

// Message that stores parameters used by L1LossLayer
message L1LossParameter {
  optional bool l2_per_location = 1 [default = false];
  optional bool l2_prescale_by_channels = 2 [default = false]; // Old style
  optional bool normalize_by_num_entries = 3 [default = false]; // if we want to normalize not by batch size, but by the number of non-NaN entries
  optional float epsilon = 4 [default = 1e-2]; // constant for smoothing near zero
  optional float plateau = 3001 [default = 0]; // L1 Errors smaller than plateau-value will result in zero loss and no gradient
}

message ReaderParameter {
    required string file = 1;
    optional uint32 num = 2 [default=1];
}

message WriterParameter {
    optional string file = 1 [default=""];
    optional string folder = 2 [default=""];
    optional string prefix = 3 [default=""];
    optional string suffix = 4 [default=""];
    optional bool normalize = 5 [default=false];
    optional float scale = 6 [default=1];
}

message MeanParameter {
  enum MeanOperation {
    ADD = 1;
    SUBTRACT = 4;
  }

  required MeanOperation operation = 1;
  optional string file = 2;
  repeated float value = 3;
  optional float mean_scale = 4 [default = 1];
  optional float input_scale = 5 [default = 1];
  optional float output_scale = 6 [default = 1];
}

message ResampleParameter {
    enum ResampleType {
        NEAREST = 1;
        LINEAR = 2;
        CUBIC = 3;
        AREA = 4;
    };
    optional bool antialias = 4 [ default = true ];
    optional uint32 width  = 1;
    optional uint32 height = 2;
    optional ResampleType type = 3 [ default = LINEAR ];
    optional float factor = 5 [ default = 1.0 ];
}

message DownsampleParameter {
  optional uint32 top_height = 1 [default = 0]; // The output height
  optional uint32 top_width = 2 [default = 0]; // The output width
}

// Message that stores parameters used by LpqLossLayer
message LpqLossParameter {
  /**
   * Legacy parameter; now applies to p-PowerLayer
   */
  optional bool l2_prescale_by_channels = 4016 [default = false];
  // if we want to normalize not by batch size, but by the number of non-NaN entries
  optional bool normalize_by_num_entries = 4017 [default = false];

  /**
   * "Shift" for p-PowerLayer. Since the Lpq layer uses this parameter
   * to avoid singularities around 0, and p is usually >=1, this parameter
   * is normally 0.
   */
  optional float p_epsilon = 4013 [default = 0];
  /**
   * "Shift" for q-PowerLayer. Since the Lpq layer uses this parameter
   * to avoid singularities around 0, and q is usually <=1, this parameter
   * is usually relevant and positive.
   */
  optional float q_epsilon = 4014 [default = 1e-2];

  /****************************
   * If one of each of the following is given, p/q are constant and the entry
   * in "pq_episode_starts_at_iter" must be 0 (else the first entry must be 0).
   *
   * If multiple values are given (the number of values must be the same for
   * each parameter), then p/q will jump at each iteration number given in
   * "pq_episode_starts_at_iter".
   ****************************
   * Example: pq_episode_starts_at_iter = {0, 1000, 500000}
   *                                  p = {1.0, 2.0, 2.0}
   *                                  q = {1.0, 1.0, 2.0}
   *
   * With these values, p/q will be 1.0/1.0 from iteration 0 to 1000, then
   * 2.0/1.0 until iteration 500000, and 2.0/2.0 afterwards.
   ****************************/
  repeated uint32 pq_episode_starts_at_iter = 4010;
  repeated float p = 4011;
  repeated float q = 4012;
}

message FlowWarpParameter {
    enum FillParameter {
        ZERO = 1;
        NOT_A_NUMBER = 2;
    }

    optional FillParameter fill_value = 1 [ default = ZERO ];
}

message AccumParameter {
  optional uint32 top_height = 1 [default = 0]; // The output height
  optional uint32 top_width = 2 [default = 0]; // The output width
  optional uint32 size_divisible_by = 3 [default = 0]; // Upscales to the minimal size divisible by the given number
  optional bool have_reference = 4 [ default = false ];
}

message YoloV2LossParameter {
  repeated float anchors = 1;
  optional int32 side = 2;
  optional int32 num_classes = 3;
  optional int32 num_object = 4;
  optional float box_scale = 5  [default = .5];
  optional float class_scale = 6  [default = .5];
  optional float noobject_scale = 7  [default = .5];
  optional float object_scale = 8  [default = .5];
  optional bool  rescore = 9  [default = true];
  optional bool  constraint = 10 [default = true];
  optional float thresh = 11 [default = 0.7];
}

message YoloV3LossParameter {
  repeated int32 anchors = 1;
  repeated int32 mask = 2;
  optional int32 side = 3;
  optional int32 num_classes = 4;
  optional int32 num_object = 5;
  optional int32 total_object = 6;
  optional float ignore_thresh = 7 [default = 0.7];
  optional float truth_thresh = 8 [default = 1.0];
  optional int32 net_w = 9;
  optional int32 net_h = 10;
}

// Message used by AugmentationParameter for describing how to generate augmentation parameters
message RandomGeneratorParameter {
  optional string rand_type = 1 [default = "uniform" ]; // can be uniform, gaussian, bernoulli
  optional bool exp = 2 [default = false ]; // after generating the random number, exponentiate it or not
  optional float mean = 4 [default = 0. ]; // mean of the random variable
  optional float spread = 5 [default = 0. ]; // half of interval length for uniform; standard deviation for gaussian
  optional float prob = 6 [default = 1.];
  optional bool apply_schedule = 7 [default = true];
  optional bool discretize = 8 [default = false]; //Discretize (Round) value from rng to INT
  optional float multiplier = 9 [default = 1.]; //Final random value will be multiplied by this. (Useful for discrete distributions)
}

message RingPadParameter {
  optional int32 axis = 1 [default = 1];
  optional int32 pad = 2 [default = 1];
}

// Message that stores parameters used by BoxOutputLayer
message BoxOutputParameter {
  // Set the cutoff threshold
  optional float fg_thr = 1 [default = 0];
  optional float iou_thr = 2 [default = 0.5];
  // Specifiy the number of coordinates
  optional string nms_type = 3 [default = "IOU"];
  repeated uint32 field_h = 4;
  repeated uint32 field_w = 5;
  repeated uint32 downsample_rate = 6;
  optional float field_whr = 7 [default = 2]; 
  optional float field_xyr = 8 [default = 2]; 
  optional uint32 max_nms_num = 9 [default = 0];
  optional uint32 max_post_nms_num = 10 [default = 0];
  optional float min_size = 11 [default = 15]; 
  optional uint32 num_param_set = 12 [default = 1];
  optional bool ringpad = 13 [default = false];
}

message YoloDetectionOutputParameter {
  // Yolo detection output layer
  optional uint32 side = 1 [default = 13];
  optional uint32 num_classes = 2 [default = 20];
  optional uint32 num_box = 3 [default = 5];
  optional uint32 coords = 4 [default = 4];
  optional float confidence_threshold = 5 [default = 0.01];
  optional float nms_threshold = 6 [default = 0.45];
  repeated float biases = 7;
  optional string label_map_file = 8;
}

message Yolov3DetectionOutputParameter {
  // Yolo detection output layer
  optional uint32 num_classes = 1 [default = 20];
  optional uint32 num_box = 2 [default = 3];
  optional float confidence_threshold = 3 [default = 0.01];
  optional float nms_threshold = 4 [default = 0.45];
  repeated float biases = 5;
  repeated uint32 anchors_scale = 6 ;
  optional uint32 mask_group_num = 7 [default = 2];
  repeated uint32 mask = 8;
}

message Yolov3Parameter{
  //Yolov3
  optional uint32 side = 1 [default = 13];
  optional uint32 num_class = 2 [default = 20];
  optional uint32 num = 3 [default = 5];
  optional float object_scale = 4 [default = 1.0];
  optional float class_scale = 5 [default = 1.0];
  optional float noobject_scale = 6 [default = 0.5];
  optional float coord_scale = 7 [default = 5.0];
  optional float thresh = 8 [default = 0.2];
  optional uint32 anchors_scale = 10 [default = 32];
  repeated float biases = 11;
  repeated uint32 mask = 12;
  optional bool use_logic_gradient = 13 [default = false];
}

// ***************** MulticoreWare_Modified - Feature: Pruning / Splicing ******************
// Convolution with pruning
message SqueezeConvolutionParameter {
  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms

  // Pad, kernel size, and stride are all given as a single value for equal
  // dimensions in all spatial dimensions, or once per spatial dimension.
  repeated uint32 pad = 3; // The padding size; defaults to 0
  repeated uint32 kernel_size = 4; // The kernel size
  repeated uint32 stride = 6; // The stride; defaults to 1
  // Factor used to dilate the kernel, (implicitly) zero-filling the resulting
  // holes. (Kernel dilation is sometimes referred to by its use in the
  // algorithme à trous from Holschneider et al. 1987.)
  repeated uint32 dilation = 24; // The dilation; defaults to 1

  // For 2D convolution only, the *_h and *_w versions may also be used to
  // specify both spatial dimensions.
  optional uint32 pad_h = 9 [default = 0]; // The padding height (2D only)
  optional uint32 pad_w = 10 [default = 0]; // The padding width (2D only)
  optional uint32 pad_type = 29 [default = 0]; //CUSTOMIZATION
  optional uint32 pad_l = 36 [default = 0]; //CUSTOMIZATION
  optional uint32 pad_r = 37 [default = 0]; //CUSTOMIZATION
  optional uint32 pad_t = 38 [default = 0]; //CUSTOMIZATION
  optional uint32 pad_b = 39 [default = 0]; //CUSTOMIZATION
  optional uint32 kernel_h = 11; // The kernel height (2D only)
  optional uint32 kernel_w = 12; // The kernel width (2D only)
  optional uint32 stride_h = 13; // The stride height (2D only)
  optional uint32 stride_w = 14; // The stride width (2D only)

  optional double input_scale = 46 [default = 1]; //CUSTOMIZATION, act as dummy param in squeeze_conv layer now
  optional double output_scale = 47 [default = 1]; //CUSTOMIZATION, act as dummy param in squeeze_conv layer now
  optional bool signed_saturate = 48 [default = false]; //CUSTOMIZATION, act as dummy param in squeeze_conv layer now

  optional uint32 group = 5 [default = 1]; // The group size for group conv

  optional FillerParameter weight_filler = 7; // The filler for the weight
  optional FillerParameter bias_filler = 8; // The filler for the bias
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 15 [default = DEFAULT];

  // The axis to interpret as "channels" when performing convolution.
  // Preceding dimensions are treated as independent inputs;
  // succeeding dimensions are treated as "spatial".
  // With (N, C, H, W) inputs, and axis == 1 (the default), we perform
  // N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
  // groups g>1) filters across the spatial axes (H, W) of the input.
  // With (N, C, D, H, W) inputs, and axis == 1, we perform
  // N independent 3D convolutions, sliding (C/g)-channels
  // filters across the spatial axes (D, H, W) of the input.
  optional int32 axis = 16 [default = 1];

  // Whether to force use of the general ND convolution, even if a specific
  // implementation for blobs of the appropriate number of spatial dimensions
  // is available. (Currently, there is only a 2D-specific convolution
  // implementation; for input blobs with num_axes != 2, this option is
  // ignored and the ND implementation will be used.)
  optional bool force_nd_im2col = 17 [default = false];
  optional float gamma = 18 [default = 0.001]; // The compress parameter of current layer
  optional float power = 19 [default = 1];
  optional float iter_stop = 20 [default = 10000];
  optional float c_rate = 21 [default = 3];
  optional FillerParameter weight_mask_filler = 22; // The filler for the weight
  optional FillerParameter bias_mask_filler = 23;   // The filler for the bias
  optional bool dynamicsplicing = 25[default = false];
  optional float splicing_rate = 26 [default = 0.0001];
}
// *****************************************************************************************

// **************** MulticoreWare_Modified - Feature: Pruning / Splicing ***************
// InnerProduct with pruning
message SqueezeInnerProductParameter {

  optional uint32 num_output = 1; // The number of outputs for the layer
  optional bool bias_term = 2 [default = true]; // whether to have bias terms
  optional FillerParameter weight_filler = 3; // The filler for the weight
  optional FillerParameter bias_filler = 4; // The filler for the bias

  // The first axis to be lumped into a single inner product computation;
  // all preceding axes are retained in the output.
  // May be negative to index from the end (e.g., -1 for the last axis).
  optional int32 axis = 5 [default = 1];
  // Specify whether to transpose the weight matrix or not.
  // If transpose == true, any operations will be performed on the transpose
  // of the weight matrix. The weight matrix itself is not going to be transposed
  // but rather the transfer flag of operations will be toggled accordingly.
  optional bool transpose = 6 [default = false];
  optional float gamma = 7 [default = 0.001]; // The compress parameter of current layer
  optional float power = 8 [default = 1];
  optional float iter_stop = 9 [default = 8000];
  optional float c_rate = 10 [default = 3];
  optional FillerParameter weight_mask_filler = 11; // The filler for the weight
  optional FillerParameter bias_mask_filler = 12;   // The filler for the bias
  optional bool dynamicsplicing = 13[default = false];
  optional float splicing_rate = 14 [default = 0.001];
}

message SmoothL1LossParameter {
  // SmoothL1Loss(x) =
  //   0.5 * (sigma * x) ** 2    -- if x < 1.0 / sigma / sigma
  //   |x| - 0.5 / sigma / sigma -- otherwise
  optional float sigma = 1 [default = 1];
}
`);

update(
`  // Whether to force use of the general ND convolution, even if a specific
  // implementation for blobs of the appropriate number of spatial dimensions
  // is available. (Currently, there is only a 2D-specific convolution
  // implementation; for input blobs with num_axes != 2, this option is
  // ignored and the ND implementation will be used.)
  optional bool force_nd_im2col = 17 [default = false];
}

message CropParameter {
`,
`  // Whether to force use of the general ND convolution, even if a specific
  // implementation for blobs of the appropriate number of spatial dimensions
  // is available. (Currently, there is only a 2D-specific convolution
  // implementation; for input blobs with num_axes != 2, this option is
  // ignored and the ND implementation will be used.)
  optional bool force_nd_im2col = 17 [default = false];
  optional uint32 hole = 117 [default = 1]; // The filter stride (equal in Y, X) // 17 in pathak22/caffe-ccnn
  optional uint32 hole_h = 118 [default = 1]; // The filter stride height // 18 in pathak22/caffe-ccnn
  optional uint32 hole_w = 119 [default = 1]; // The filter stride width // 19s in pathak22/caffe-ccnn
}

message CropParameter {
`);

update(
`    SLICE = 33;
    TANH = 23;
    WINDOW_DATA = 24;
    THRESHOLD = 31;
  }
`,
`    SLICE = 33;
    TANH = 23;
    WINDOW_DATA = 24;
    THRESHOLD = 31;
    IMAGE_SEG_DATA = 139; // 39 in cdmh/deeplab-public
  }
`);

fs.writeFileSync(file, data, 'utf-8');
