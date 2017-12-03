import numpy as np
import cv2

scale_subregion = float(3) / 4
scale_mask = float(1)/(scale_subregion*4)
def calculate_iou(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    img_or = cv2.bitwise_or(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(img_or)
    iou = float(float(j)/float(i))
    return iou


def calculate_overlapping(img_mask, gt_mask):
    gt_mask *= 1.0
    img_and = cv2.bitwise_and(img_mask, gt_mask)
    j = np.count_nonzero(img_and)
    i = np.count_nonzero(gt_mask)
    overlap = float(float(j)/float(i))
    return overlap


def follow_iou(gt_masks, region_mask, classes_gt_objects, class_object, last_matrix):
    results = np.zeros([np.size(array_classes_gt_objects), 1])
    for k in range(np.size(classes_gt_objects)):
        if classes_gt_objects[k] == class_object:
            gt_mask = gt_masks[:, :, k]
            iou = calculate_iou(region_mask, gt_mask)
            results[k] = iou
    index = np.argmax(results)
    new_iou = results[index]
    iou = last_matrix[index]
    return iou, new_iou, results, index

# Auto find the max bounding box in the region image
def find_max_bounding_box(gt_masks, region_mask, classes_gt_objects, class_object):
    _, _, n = gt_masks.shape
    max_iou = 0.0
    for k in range(n):
        if classes_gt_objects[k] != class_object:
            continue
        gt_mask = gt_masks[:,:,k]
        iou = calculate_iou(region_mask, gt_mask)
        if max_iou < iou:
            max_iou = iou
    return max_iou

def get_crop_image_and_mask(original_shape, offset, region_image, size_mask, action):
    r"""crop the the image according to action
    
    Args:
        original_shape: shape of original image (H x W)
        offset: the current image's left-top coordinate base on the original image
        region_image: the image to be cropped
        size_mask: the size of region_image
        action: the action choose by agent. can be 1,2,3,4,5.
        
    Returns:
        offset: the cropped image's left-top coordinate base on original image
        region_image: the cropped image
        size_mask: the size of the cropped image
        region_mask: the masked image which mask cropped region and has same size with original image
    
    """
    
    
    region_mask = np.zeros(original_shape) # mask at original image 
    size_mask = (int(size_mask[0] * scale_subregion), int(size_mask[1] * scale_subregion)) # the size of croped image
    if action == 1:
        offset_aux = (0, 0)
    elif action == 2:
        offset_aux = (0, int(size_mask[1] * scale_mask))
        offset = (offset[0], offset[1] + int(size_mask[1] * scale_mask))
    elif action == 3:
        offset_aux = (int(size_mask[0] * scale_mask), 0)
        offset = (offset[0] + int(size_mask[0] * scale_mask), offset[1])
    elif action == 4:
        offset_aux = (int(size_mask[0] * scale_mask), 
                      int(size_mask[1] * scale_mask))
        offset = (offset[0] + int(size_mask[0] * scale_mask),
                  offset[1] + int(size_mask[1] * scale_mask))
    elif action == 5:
        offset_aux = (int(size_mask[0] * scale_mask / 2),
                      int(size_mask[0] * scale_mask / 2))
        offset = (offset[0] + int(size_mask[0] * scale_mask / 2),
                  offset[1] + int(size_mask[0] * scale_mask / 2))
    region_image = region_image[offset_aux[0]:offset_aux[0] + size_mask[0],
                   offset_aux[1]:offset_aux[1] + size_mask[1]]
    region_mask[offset[0]:offset[0] + size_mask[0], offset[1]:offset[1] + size_mask[1]] = 1
    return offset, region_image, size_mask, region_mask
