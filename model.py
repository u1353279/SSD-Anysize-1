from torch import nn
import torch.nn.functional as F
from math import sqrt
from itertools import product as product
import torchvision
import numpy as np

from utils import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AuxiliaryConvolutions(nn.Module):
    """
    Additional convolutions to produce higher-level feature maps.
    """

    def __init__(self, in_shape):
        super(AuxiliaryConvolutions, self).__init__()

        self.in_depth = in_shape[0]
        self.in_dims = in_shape[1:]

        self.SSDconv1_1 = nn.Conv2d(self.in_depth, 256, kernel_size=1, padding=0)  # stride = 1, by default
        self.SSDconv1_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.SSDconv2_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.SSDconv2_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # dim. reduction because stride > 1

        self.SSDconv3_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.SSDconv3_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)

        self.SSDconv4_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.SSDconv4_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  # dim. reduction because padding = 0
        
        # Get the info required to make the prior boxes later
        self.get_construction_info()

        # Initialize convolutions' parameters
        self.init_conv2d()


    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, backbone_out2_feats):
        """
        Forward propagation.

        :param backbone_out2_feats: lower-level backbone_out2 feature map, a tensor of dimensions (N, DEPTH, SIZE, SIZE)
        :return: higher-level feature maps SSDconv1_2, SSDconv2_2, SSDconv3_2, and SSDconv4_2
        """
        out = F.relu(self.SSDconv1_1(backbone_out2_feats))
        out = F.relu(self.SSDconv1_2(out)) 
        SSDconv1_2_feats = out

        out = F.relu(self.SSDconv2_1(out))
        out = F.relu(self.SSDconv2_2(out))
        SSDconv2_2_feats = out

        out = F.relu(self.SSDconv3_1(out))
        out = F.relu(self.SSDconv3_2(out))
        SSDconv3_2_feats = out
        
        out = F.relu(self.SSDconv4_1(out))
        SSDconv4_2_feats = F.relu(self.SSDconv4_2(out))

        return SSDconv1_2_feats, SSDconv2_2_feats, SSDconv3_2_feats, SSDconv4_2_feats

    def get_construction_info(self):
        mock_image = np.ones(self.in_dims)
        mock_image = np.dstack([mock_image] * self.in_depth)
        mock_image = mock_image.transpose()
        mock_image = mock_image[np.newaxis, ...]
        mock_image = torch.from_numpy(mock_image).float()

        f1, f2, f3, f4 = self.forward(mock_image)

        # feature format is [DEPTH, IMSIZE, IMSIZE]
        self.SSDconv1_2_out_size = f1.shape[-1]
        self.SSDconv1_2_out_depth = f1.shape[1]
        self.SSDconv2_2_out_size = f2.shape[-1]
        self.SSDconv2_2_out_depth = f2.shape[1]
        self.SSDconv3_2_out_size = f3.shape[-1]
        self.SSDconv3_2_out_depth = f3.shape[1]
        self.SSDconv4_2_out_size = f4.shape[-1]
        self.SSDconv4_2_out_depth = f4.shape[1]


class PredictionConvolutions(nn.Module):
    """
    Convolutions to predict class scores and bounding boxes using lower and higher-level feature maps.

    The bounding boxes (locations) are predicted as encoded offsets w.r.t each of the 8732 prior (default) boxes.
    See 'cxcy_to_gcxgcy' in utils.py for the encoding definition.

    The class scores represent the scores of each object class in each of the 8732 bounding boxes located.
    A high score for 'background' = no object.
    """

    def __init__(self, n_classes, out1_shape, out2_shape):
        """
        :param n_classes: number of different types of objects
        """
        super(PredictionConvolutions, self).__init__()

        self.n_classes = n_classes
        self.out1_depth = out1_shape[0]
        self.out2_depth = out2_shape[0]

        # Number of prior-boxes we are considering per position in each feature map.
        # 4 prior-boxes implies we use 4 different aspect ratios, etc.
        # TODO: Adjust priors per size, maybe not necessary
        n_boxes = {'backbone_out1': 4,
                   'backbone_out2': 6,
                   'SSDconv1_2': 6,
                   'SSDconv2_2': 6,
                   'SSDconv3_2': 4,
                   'SSDconv4_2': 4}
        # 

        # Localization prediction convolutions (predict offsets w.r.t prior-boxes)
        # self.loc_backbone_out1 = nn.Conv2d(512, n_boxes['backbone_out1'] * 4, kernel_size=3, padding=1)
        self.loc_backbone_out1 = nn.Conv2d(self.out1_depth, n_boxes['backbone_out1'] * 4, kernel_size=3, padding=1)
        self.loc_backbone_out2 = nn.Conv2d(self.out2_depth, n_boxes['backbone_out2'] * 4, kernel_size=3, padding=1)
        self.loc_SSDconv1_2 = nn.Conv2d(512, n_boxes['SSDconv1_2'] * 4, kernel_size=3, padding=1)
        self.loc_SSDconv2_2 = nn.Conv2d(256, n_boxes['SSDconv2_2'] * 4, kernel_size=3, padding=1)
        self.loc_SSDconv3_2 = nn.Conv2d(256, n_boxes['SSDconv3_2'] * 4, kernel_size=3, padding=1)
        self.loc_SSDconv4_2 = nn.Conv2d(256, n_boxes['SSDconv4_2'] * 4, kernel_size=3, padding=1)

        # Class prediction convolutions (predict classes in localization boxes)
        self.cl_backbone_out1 = nn.Conv2d(self.out1_depth, n_boxes['backbone_out1'] * n_classes, kernel_size=3, padding=1)
        self.cl_backbone_out2 = nn.Conv2d(self.out2_depth, n_boxes['backbone_out2'] * n_classes, kernel_size=3, padding=1)
        self.cl_SSDconv1_2 = nn.Conv2d(512, n_boxes['SSDconv1_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_SSDconv2_2 = nn.Conv2d(256, n_boxes['SSDconv2_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_SSDconv3_2 = nn.Conv2d(256, n_boxes['SSDconv3_2'] * n_classes, kernel_size=3, padding=1)
        self.cl_SSDconv4_2 = nn.Conv2d(256, n_boxes['SSDconv4_2'] * n_classes, kernel_size=3, padding=1)

        # Initialize convolutions' parameters
        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, backbone_out1_feats, backbone_out2_feats, SSDconv1_2_feats, SSDconv2_2_feats, SSDconv3_2_feats, SSDconv4_2_feats):
        """
        Forward propagation.

        :param backbone_out1_feats: backbone_out1 feature map, a tensor of dimensions (N, 512, 38, 38)
        :param backbone_out2_feats: backbone_out2 feature map, a tensor of dimensions (N, 1024, 19, 19)
        :param SSDconv1_2_feats: SSDconv1_2 feature map, a tensor of dimensions (N, 512, 10, 10)
        :param SSDconv2_2_feats: SSDconv2_2 feature map, a tensor of dimensions (N, 256, 5, 5)
        :param SSDconv3_2_feats: SSDconv3_2 feature map, a tensor of dimensions (N, 256, 3, 3)
        :param SSDconv4_2_feats: SSDconv4_2 feature map, a tensor of dimensions (N, 256, 1, 1)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        batch_size = backbone_out1_feats.size(0)

        # Predict localization boxes' bounds (as offsets w.r.t prior-boxes)
        l_backbone_out1 = self.loc_backbone_out1(backbone_out1_feats)  # (N, 16, 38, 38)
        l_backbone_out1 = l_backbone_out1.permute(0, 2, 3,1).contiguous()  # (N, 38, 38, 16), to match prior-box order (after .view())
        # (.contiguous() ensures it is stored in a contiguous chunk of memory, needed for .view() below)
        l_backbone_out1 = l_backbone_out1.view(batch_size, -1, 4)  # (N, 5776, 4), there are a total 5776 boxes on this feature map

        l_backbone_out2 = self.loc_backbone_out2(backbone_out2_feats)  # (N, 24, 19, 19)
        l_backbone_out2 = l_backbone_out2.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 24)
        l_backbone_out2 = l_backbone_out2.view(batch_size, -1, 4)  # (N, 2166, 4), there are a total 2116 boxes on this feature map

        l_SSDconv1_2 = self.loc_SSDconv1_2(SSDconv1_2_feats)  # (N, 24, 10, 10)
        l_SSDconv1_2 = l_SSDconv1_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 24)
        l_SSDconv1_2 = l_SSDconv1_2.view(batch_size, -1, 4)  # (N, 600, 4)

        l_SSDconv2_2 = self.loc_SSDconv2_2(SSDconv2_2_feats)  # (N, 24, 5, 5)
        l_SSDconv2_2 = l_SSDconv2_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 24)
        l_SSDconv2_2 = l_SSDconv2_2.view(batch_size, -1, 4)  # (N, 150, 4)

        l_SSDconv3_2 = self.loc_SSDconv3_2(SSDconv3_2_feats)  # (N, 16, 3, 3)
        l_SSDconv3_2 = l_SSDconv3_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 16)
        l_SSDconv3_2 = l_SSDconv3_2.view(batch_size, -1, 4)  # (N, 36, 4)

        l_SSDconv4_2 = self.loc_SSDconv4_2(SSDconv4_2_feats)  # (N, 16, 1, 1)
        l_SSDconv4_2 = l_SSDconv4_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 16)
        l_SSDconv4_2 = l_SSDconv4_2.view(batch_size, -1, 4)  # (N, 4, 4)

        # Predict classes in localization boxes
        c_backbone_out1 = self.cl_backbone_out1(backbone_out1_feats)  # (N, 4 * n_classes, 38, 38)
        c_backbone_out1 = c_backbone_out1.permute(0, 2, 3,
                                      1).contiguous()  # (N, 38, 38, 4 * n_classes), to match prior-box order (after .view())
        c_backbone_out1 = c_backbone_out1.view(batch_size, -1,
                                   self.n_classes)  # (N, 5776, n_classes), there are a total 5776 boxes on this feature map

        c_backbone_out2 = self.cl_backbone_out2(backbone_out2_feats)  # (N, 6 * n_classes, 19, 19)
        c_backbone_out2 = c_backbone_out2.permute(0, 2, 3, 1).contiguous()  # (N, 19, 19, 6 * n_classes)
        c_backbone_out2 = c_backbone_out2.view(batch_size, -1,
                               self.n_classes)  # (N, 2166, n_classes), there are a total 2116 boxes on this feature map

        c_SSDconv1_2 = self.cl_SSDconv1_2(SSDconv1_2_feats)  # (N, 6 * n_classes, 10, 10)
        c_SSDconv1_2 = c_SSDconv1_2.permute(0, 2, 3, 1).contiguous()  # (N, 10, 10, 6 * n_classes)
        c_SSDconv1_2 = c_SSDconv1_2.view(batch_size, -1, self.n_classes)  # (N, 600, n_classes)


        c_SSDconv2_2 = self.cl_SSDconv2_2(SSDconv2_2_feats)  # (N, 6 * n_classes, 5, 5)
        c_SSDconv2_2 = c_SSDconv2_2.permute(0, 2, 3, 1).contiguous()  # (N, 5, 5, 6 * n_classes)
        c_SSDconv2_2 = c_SSDconv2_2.view(batch_size, -1, self.n_classes)  # (N, 150, n_classes)

        c_SSDconv3_2 = self.cl_SSDconv3_2(SSDconv3_2_feats)  # (N, 4 * n_classes, 3, 3)
        c_SSDconv3_2 = c_SSDconv3_2.permute(0, 2, 3, 1).contiguous()  # (N, 3, 3, 4 * n_classes)
        c_SSDconv3_2 = c_SSDconv3_2.view(batch_size, -1, self.n_classes)  # (N, 36, n_classes)

        c_SSDconv4_2 = self.cl_SSDconv4_2(SSDconv4_2_feats)  # (N, 4 * n_classes, 1, 1)
        c_SSDconv4_2 = c_SSDconv4_2.permute(0, 2, 3, 1).contiguous()  # (N, 1, 1, 4 * n_classes)
        c_SSDconv4_2 = c_SSDconv4_2.view(batch_size, -1, self.n_classes)  # (N, 4, n_classes)

        # A total of 8732 boxes
        # Concatenate in this specific order (i.e. must match the order of the prior-boxes)
        locs = torch.cat([l_backbone_out1, l_backbone_out2, l_SSDconv1_2, l_SSDconv2_2, l_SSDconv3_2, l_SSDconv4_2], dim=1)  # (N, 8732, 4)
        classes_scores = torch.cat([c_backbone_out1, c_backbone_out2, c_SSDconv1_2, c_SSDconv2_2, c_SSDconv3_2, c_SSDconv4_2],
                                   dim=1)  # (N, 8732, n_classes)

        return locs, classes_scores


class SSD300(nn.Module):
    """
    The SSD300 network - encapsulates the base VGG network, auxiliary, and prediction convolutions.
    """

    def __init__(self, backbone, n_classes):
        super(SSD300, self).__init__()

        self.n_classes = n_classes
        self.base = backbone
        self.out1_shape = self.base.out_shape_1
        self.out2_shape = self.base.out_shape_2

        self.aux_convs = AuxiliaryConvolutions(self.out2_shape)
        self.pred_convs = PredictionConvolutions(n_classes, self.out1_shape, self.out2_shape)

        # Since lower level features (backbone_out1_feats) have considerably larger scales, we take the L2 norm and rescale
        # Rescale factor is initially set at 20, but is learned for each channel during back-prop

        # self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in backbone_out1_feats
        out1_depth = self.out1_shape[0]
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, out1_depth, 1, 1))
        nn.init.constant_(self.rescale_factors, 20)

        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

    def forward(self, image):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: 8732 locations and class scores (i.e. w.r.t each prior box) for each image
        """
        # Run VGG base network convolutions (lower level feature map generators)
        backbone_out1_feats, backbone_out2_feats = self.base(image)

        # Rescale backbone_out1 after L2 norm
        norm = backbone_out1_feats.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        backbone_out1_feats = backbone_out1_feats / norm  # (N, 512, 38, 38)
        backbone_out1_feats = backbone_out1_feats * self.rescale_factors  # (N, 512, 38, 38)
        # (PyTorch autobroadcasts singleton dimensions during arithmetic)

        # Run auxiliary convolutions (higher level feature map generators)
        SSDconv1_2_feats, SSDconv2_2_feats, SSDconv3_2_feats, SSDconv4_2_feats = self.aux_convs(backbone_out2_feats)  # (N, 512, 10, 10),  (N, 256, 5, 5), (N, 256, 3, 3), (N, 256, 1, 1)

        # Run prediction convolutions (predict offsets w.r.t prior-boxes and classes in each resulting localization box)
        locs, classes_scores = self.pred_convs(backbone_out1_feats, backbone_out2_feats, SSDconv1_2_feats, SSDconv2_2_feats, SSDconv3_2_feats,
                                               SSDconv4_2_feats)  # (N, 8732, 4), (N, 8732, n_classes)

        return locs, classes_scores

    def create_prior_boxes(self):
        """
        Create the 8732 prior (default) boxes for the SSD300, as defined in the paper.

        :return: prior boxes in center-size coordinates, a tensor of dimensions (8732, 4)
        """

        fmap_dims = {'backbone_out1': self.base.out_shape_1[-1], # out_shape is in [DEPTH, SIZE, SIZE], only need 1 size dim
                     'backbone_out2': self.base.out_shape_2[-1],
                     'SSDconv1_2': self.aux_convs.SSDconv1_2_out_size,
                     'SSDconv2_2': self.aux_convs.SSDconv2_2_out_size,
                     'SSDconv3_2': self.aux_convs.SSDconv3_2_out_size,
                     'SSDconv4_2': self.aux_convs.SSDconv4_2_out_size}

        obj_scales = {'backbone_out1': 0.1,
                      'backbone_out2': 0.2,
                      'SSDconv1_2': 0.375,
                      'SSDconv2_2': 0.55,
                      'SSDconv3_2': 0.725,
                      'SSDconv4_2': 0.9}

        aspect_ratios = {'backbone_out1': [1., 2., 0.5],
                         'backbone_out2': [1., 2., 3., 0.5, .333],
                         'SSDconv1_2': [1., 2., 3., 0.5, .333],
                         'SSDconv2_2': [1., 2., 3., 0.5, .333],
                         'SSDconv3_2': [1., 2., 0.5],
                         'SSDconv4_2': [1., 2., 0.5]}

        fmaps = list(fmap_dims.keys())

        prior_boxes = []

        for k, fmap in enumerate(fmaps):
            for i in range(fmap_dims[fmap]):
                for j in range(fmap_dims[fmap]):
                    cx = (j + 0.5) / fmap_dims[fmap]
                    cy = (i + 0.5) / fmap_dims[fmap]

                    for ratio in aspect_ratios[fmap]:
                        prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

                        # For an aspect ratio of 1, use an additional prior whose scale is the geometric mean of the
                        # scale of the current feature map and the scale of the next feature map
                        if ratio == 1.:
                            try:
                                additional_scale = sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            # For the last feature map, there is no "next" feature map
                            except IndexError:
                                additional_scale = 1.
                            prior_boxes.append([cx, cy, additional_scale, additional_scale])

        prior_boxes = torch.FloatTensor(prior_boxes).to(device)  # (8732, 4)
        prior_boxes.clamp_(0, 1)  # (8732, 4)

        return prior_boxes

    def detect_objects(self, predicted_locs, predicted_scores, min_score, max_overlap, top_k):
        """
        Decipher the 8732 locations and class scores (output of ths SSD300) to detect objects.

        For each class, perform Non-Maximum Suppression (NMS) on boxes that are above a minimum threshold.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param min_score: minimum threshold for a box to be considered a match for a certain class
        :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via NMS
        :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
        :return: detections (boxes, labels, and scores), lists of length batch_size
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, n_classes)

        # Lists to store final predicted boxes, labels, and scores for all images
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        for i in range(batch_size):
            # Decode object coordinates from the form we regressed predicted boxes to
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4), these are fractional pt. coordinates

            # Lists to store boxes and scores for this image
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            # Check for each class
            for c in range(1, self.n_classes):
                # Keep only predicted boxes and scores where scores for this class are above the minimum score
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  # torch.uint8 (byte) tensor, for indexing
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  # (n_qualified), n_min_score <= 8732
                class_decoded_locs = decoded_locs[score_above_min_score]  # (n_qualified, 4)

                # Sort predicted boxes and scores by scores
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True)  # (n_qualified), (n_min_score)
                class_decoded_locs = class_decoded_locs[sort_ind]  # (n_min_score, 4)

                # Find the overlap between predicted boxes
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  # (n_qualified, n_min_score)

                # Non-Maximum Suppression (NMS)

                # A torch.uint8 (byte) tensor to keep track of which predicted boxes to suppress
                # 1 implies suppress, 0 implies don't suppress
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  # (n_qualified)

                # Consider each box in order of decreasing scores
                for box in range(class_decoded_locs.size(0)):
                    # If this box is already marked for suppression
                    if suppress[box] == 1:
                        continue

                    # Suppress boxes whose overlaps (with this box) are greater than maximum overlap
                    # Find such boxes and update suppress indices
                    suppress = torch.max(suppress, overlap[box] > max_overlap)
                    # The max operation retains previously suppressed boxes, like an 'OR' operation

                    # Don't suppress this box, even though it has an overlap of 1 with itself
                    suppress[box] = 0

                # Store only unsuppressed boxes for this class
                image_boxes.append(class_decoded_locs[1 - suppress])
                image_labels.append(torch.LongTensor((1 - suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[1 - suppress])

            # If no object in any class is found, store a placeholder for 'background'
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

            # Concatenate into single tensors
            image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
            image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
            image_scores = torch.cat(image_scores, dim=0)  # (n_objects)
            n_objects = image_scores.size(0)

            # Keep only the top k objects
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  # (top_k)
                image_boxes = image_boxes[sort_ind][:top_k]  # (top_k, 4)
                image_labels = image_labels[sort_ind][:top_k]  # (top_k)

            # Append to lists that store predicted boxes and scores for all images
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  # lists of length batch_size


class MultiBoxLoss(nn.Module):
    """
    The MultiBox loss, a loss function for object detection.

    This is a combination of:
    (1) a localization loss for the predicted locations of the boxes, and
    (2) a confidence loss for the predicted class scores.
    """

    def __init__(self, priors_cxcy, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(MultiBoxLoss, self).__init__()
        self.priors_cxcy = priors_cxcy
        self.priors_xy = cxcy_to_xy(priors_cxcy)
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, predicted_locs, predicted_scores, boxes, labels):
        """
        Forward propagation.

        :param predicted_locs: predicted locations/boxes w.r.t the 8732 prior boxes, a tensor of dimensions (N, 8732, 4)
        :param predicted_scores: class scores for each of the encoded locations/boxes, a tensor of dimensions (N, 8732, n_classes)
        :param boxes: true  object bounding boxes in boundary coordinates, a list of N tensors
        :param labels: true object labels, a list of N tensors
        :return: multibox loss, a scalar
        """
        batch_size = predicted_locs.size(0)
        n_priors = self.priors_cxcy.size(0)
        n_classes = predicted_scores.size(2)

        assert n_priors == predicted_locs.size(1) == predicted_scores.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        true_classes = torch.zeros((batch_size, n_priors), dtype=torch.long).to(device)  # (N, 8732)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i],  self.priors_xy)  # (n_objects, 8732)

            # For each prior, find the object that has the maximum overlap
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)  # (8732)

            # We don't want a situation where an object is not represented in our positive (non-background) priors -
            # 1. An object might not be the best object for all priors, and is therefore not in object_for_each_prior.
            # 2. All priors with the object may be assigned as background based on the threshold (0.5).

            # To remedy this -
            # First, find the prior that has the maximum overlap for each object.
            _, prior_for_each_object = overlap.max(dim=1)  # (N_o)

            # Then, assign each object to the corresponding maximum-overlap-prior. (This fixes 1.)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects)).to(device)

            # To ensure these priors qualify, artificially give them an overlap of greater than 0.5. (This fixes 2.)
            overlap_for_each_prior[prior_for_each_object] = 1.

            # Labels for each prior
            label_for_each_prior = labels[i][object_for_each_prior]  # (8732)
            # Set priors whose overlaps with objects are less than the threshold to be background (no object)
            label_for_each_prior[overlap_for_each_prior < self.threshold] = 0  # (8732)

            # Store
            true_classes[i] = label_for_each_prior

            # Encode center-size object coordinates into the form we regressed predicted boxes to
            true_locs[i] = cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_for_each_prior]), self.priors_cxcy)  # (8732, 4)

        # Identify priors that are positive (object/non-background)
        positive_priors = true_classes != 0  # (N, 8732)

        # LOCALIZATION LOSS

        # Localization loss is computed only over positive (non-background) priors
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])  # (), scalar

        # Note: indexing with a torch.uint8 (byte) tensor flattens the tensor when indexing is across multiple dimensions (N & 8732)
        # So, if predicted_locs has the shape (N, 8732, 4), predicted_locs[positive_priors] will have (total positives, 4)

        # CONFIDENCE LOSS

        # Confidence loss is computed over positive priors and the most difficult (hardest) negative priors in each image
        # That is, FOR EACH IMAGE,
        # we will take the hardest (neg_pos_ratio * n_positives) negative priors, i.e where there is maximum loss
        # This is called Hard Negative Mining - it concentrates on hardest negatives in each image, and also minimizes pos/neg imbalance

        # Number of positive and hard-negative priors per image
        n_positives = positive_priors.sum(dim=1)  # (N)
        n_hard_negatives = self.neg_pos_ratio * n_positives  # (N)

        # First, find the loss for all priors
        conf_loss_all = self.cross_entropy(predicted_scores.view(-1, n_classes), true_classes.view(-1))  # (N * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, n_priors)  # (N, 8732)

        # We already know which priors are positive
        conf_loss_pos = conf_loss_all[positive_priors]  # (sum(n_positives))

        # Next, find which priors are hard-negative
        # To do this, sort ONLY negative priors in each image in order of decreasing loss and take top n_hard_negatives
        conf_loss_neg = conf_loss_all.clone()  # (N, 8732)
        conf_loss_neg[positive_priors] = 0.  # (N, 8732), positive priors are ignored (never in top n_hard_negatives)
        conf_loss_neg, _ = conf_loss_neg.sort(dim=1, descending=True)  # (N, 8732), sorted by decreasing hardness
        hardness_ranks = torch.LongTensor(range(n_priors)).unsqueeze(0).expand_as(conf_loss_neg).to(device)  # (N, 8732)
        hard_negatives = hardness_ranks < n_hard_negatives.unsqueeze(1)  # (N, 8732)
        conf_loss_hard_neg = conf_loss_neg[hard_negatives]  # (sum(n_hard_negatives))

        # As in the paper, averaged over positive priors only, although computed over both positive and hard-negative priors
        conf_loss = (conf_loss_hard_neg.sum() + conf_loss_pos.sum()) / n_positives.sum().float()  # (), scalar

        # TOTAL LOSS

        return conf_loss + self.alpha * loc_loss
