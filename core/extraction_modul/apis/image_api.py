import itertools
from collections import defaultdict

from ._base_api_ import TransformationStrategy
from ..extraction_model import PDF_Extraction
from core.config import IMAGE_DIRECTORY, TMP_DIRECTORY
import os
from ..datamodels.image_models import Image
from ..datamodels.internal_models import TextBlock
import core.detection_models.img_detection as img
from core.detection_models.detection import predict_table_boundaries, in_json
from typing import List, Tuple
import fitz
from functools import lru_cache


class ImageStrategy(TransformationStrategy):
    ''' An Agent performing all necessary tasks for the extraction and transformation of the image_file. '''
    IMAGE_DETECTION_MODEL = None

    def __init__(self):
        super().__init__()
        ImageStrategy.IMAGE_DETECTION_MODEL = img.load_img_detection_model()

    def postprocess_data(self, data: PDF_Extraction) -> None:
        self.check_complete_images(data)

    def check_complete_images(self, data):
        res = []
        for image in data.images:
            if image.coordinatesOfPicture != [0, 0, 1, 1] and image.image_file is not None:
                res.append(image)
        print("ok")
        data.images = res

    def preprocess_data(self, data: PDF_Extraction) -> None:
        image_descriptions = self.identify_image_descriptions(data)

        for imageBlock in image_descriptions:
            image = Image(imageBlock)
            data.images.append(image)

        self.identify_surrounding_text_blocks(data)

        self.set_predicted_boundaries(data)
        self.set_calculated_boundaries(data)
        self.get_boundaries_for_images(data)

        self.check_images(data)

    def process_data(self, data: PDF_Extraction) -> None:
        ''' Processes the found images. '''
        for image in data.images:
            if image.is_image:
                image.get_image_of_image(data.pages)
            else:
                for tBlock in image.textBlocksOfImage:
                    tBlock.isPartOfImage = False

    def is_inside(self, image):

        for boundary in image.predicted_boundaries:
            if boundary["x1"] <= image.posX1 < boundary["x2"] or boundary["x1"] <= image.posX2 < boundary["x2"]:
                if boundary["y1"] <= image.posY1 < boundary["y2"] or boundary["y1"] <= image.posY2 < boundary["y2"]:
                    return boundary, True
        else:
            return [], False

    def get_boundaries_for_images(self, data: PDF_Extraction):
        taken_boundaries = []
        zwerg = defaultdict(list)
        images = []
        zwerg_ = []
        for image in data.images:
            if not image.check_multiple_existence(data.images):
                images.append(image)
            else:
                zwerg_.append(image)

        for image in zwerg_:
            _, in_boundary = self.is_inside(image)
            if in_boundary:
                images.append(image)

        data.images = images

        for image in data.images:
            has_prediction: bool = len(image.predicted_boundaries) > 0
            has_multiple_predictions: bool = len(image.predicted_boundaries) > 1
            boundary, in_boundary = self.is_inside(image)


            if has_prediction:
                if has_multiple_predictions:
                    if not in_boundary:
                        zwerg[image.pageNum].append(image)
                    if in_boundary and not boundary in taken_boundaries :
                        image.posX1 = min(image.posX1, boundary["x1"])
                        image.posY1 = min(image.posY1, boundary['y1'])
                        image.posX2 = max(image.posX2, boundary['x2'])
                        image.posY2 = max(image.posY2, boundary['y2'])
                        image.coordinatesOfPicture = (image.posX1, image.posY1, image.posX2, image.posY2)
                        image.in_prediction = True
                        taken_boundaries.append(boundary)
                else:
                    bbox = image.predicted_boundaries[0]
                    image.posX1 = min(image.posX1, bbox["x1"])
                    image.posY1 = min(image.posY1, bbox['y1'])
                    image.posX2 = max(image.posX2, bbox['x2'])
                    image.posY2 = max(image.posY2, bbox['y2'])
                    image.coordinatesOfPicture = (image.posX1, image.posY1, image.posX2, image.posY2)
            else:
                image.posX1 = min(image.posX1, image.calculated_boundaries[0])
                image.posY1 = min(image.posY1, image.calculated_boundaries[1])
                image.posX2 = max(image.posX2, image.calculated_boundaries[2])
                image.posY2 = max(image.posY2, image.calculated_boundaries[3])
                image.coordinatesOfPicture = (image.posX1, image.posY1, image.posX2, image.posY2)

        for _, images in zwerg.items():

            def get_min_distance(image, tBlock):
                dist_1 = ((tBlock.posX2 - image['x2']) ** 2 + (tBlock.posY1 - image['y1']) ** 2) ** (1 / 2)
                dist_2 = ((tBlock.posX2 - image['x2']) ** 2 + (tBlock.posY2 - image['y2']) ** 2) ** (1 / 2)
                dist_3 = ((tBlock.posX1 - image['x1']) ** 2 + (tBlock.posY1 - image['y1']) ** 2) ** (1 / 2)
                dist_4 = ((tBlock.posX1 - image['x1']) ** 2 + (tBlock.posY2 - image['y2']) ** 2) ** (1 / 2)
                return min([dist_1, dist_2, dist_3, dist_4])

            # Calculate the distances to each image

            for image in images:
                # get the overall smallest distance
                prevDistance = 9999
                bestFit = None
                # Get the next prediction to the description
                for boundary in image.predicted_boundaries:
                    if boundary in taken_boundaries:
                        continue

                    distance = get_min_distance(boundary, image)

                    if prevDistance > distance:
                        bestFit = boundary
                        prevDistance = distance

                if bestFit:
                    image.posX1 = min(image.posX1, bestFit['x1'])
                    image.posY1 = min(image.posY1, bestFit['y1'])
                    image.posX2 = max(image.posX2, bestFit['x2'])
                    image.posY2 = max(image.posY2, bestFit['y2'])
                    image.coordinatesOfPicture = (image.posX1, image.posY1, image.posX2, image.posY2)
                    taken_boundaries.append(bestFit)

    def set_predicted_boundaries(self, data: PDF_Extraction):
        """ Identifies the boundaries of an image. """
        for image in data.images:
            page = image.pageNum
            path = self.page_as_image(data, page)

            boundaries = self.get_boundaries(ImageStrategy.IMAGE_DETECTION_MODEL,
                                             path,
                                             page)

            image.predicted_boundaries = boundaries

    def page_as_image(self, data: PDF_Extraction, page: fitz.Page):

        path_to_image = os.path.join(TMP_DIRECTORY, f"imgs/{page}.png")
        doc = data.document
        page = doc.load_page(page)
        pix = page.get_pixmap()
        pix.writePNG(path_to_image)
        return path_to_image

    #@lru_cache()
    def get_boundaries(self, model, path, page):
        prediction_res = predict_table_boundaries(model, path)
        return in_json(prediction_res, path, page)

    def check_images(self, data: PDF_Extraction) -> List[Image]:
        '''
        Checks after preProcessing if a previous found image_file is indeed an image_file
        :return:
        '''
        res = []
        for image in data.images:
            image.checkIsImage(data.images)
            if image.is_image:
                res.append(image)
        data.images = res

    def set_calculated_boundaries(self, data: PDF_Extraction) -> None:
        for image in data.images:
            self.setCoordinatesOfPicture(image)

    def setCoordinatesOfPicture(self, image):
        '''
        A description of the idea of this algorithmn can be found in the documentation.
        :return:
        '''

        # TBlockR := Textblock Right To Description Block
        TBlockL: TextBlock = self.getTextNextToDescriptionBlock(image, 0)  # Evtl hier
        TBlockR: TextBlock = self.getTextNextToDescriptionBlock(image, 1)
        TBlockO: TextBlock = self.getTextNextToDescriptionBlock(image, 2)
        TBlockU: TextBlock = self.getTextNextToDescriptionBlock(image, 3)

        # pictureIsBellow: bool = imageExtractor.HEIGHT_OF_PAGE - self.posY2 > self.posY1 - TBlockO.posY2

        posX1 = TBlockL.posX2 if TBlockL is not None else Image.MIN_WIDTH_OF_PAGE
        posX2 = TBlockR.posX1 if TBlockR is not None else Image.MAX_WIDTH_OF_PAGE
        posY1 = TBlockO.posY2 if TBlockO is not None else Image.MIN_HEIGHT_OF_PAGE
        posY2 = TBlockU.posY1 if TBlockU is not None else Image.MAX_HEIGHT_OF_PAGE

        image.calculated_boundaries = [posX1, posY1, posX2, posY2]

    def getTextNextToDescriptionBlock(self, image, side: int = 0):
        surroundingBlocks = image.surroundingDataBlocks
        surroundingBlocks.append(image)

        distanceBetweenBlocks = lambda block1, block2: image._getFirstCoordinate(block1,
                                                                                 side) - image._getSecondCoordinate(
            block2, side)
        focusedBlocks: List[TextBlock] = image._getOnlyRelevantTextBlocks(surroundingBlocks, side)
        textBlockNextToDescriptionBlock = None
        for textBlock in focusedBlocks:

            distanceToNextTextBlock = distanceBetweenBlocks(image, textBlock)
            if 0 < distanceToNextTextBlock:
                if textBlockNextToDescriptionBlock is None:
                    textBlockNextToDescriptionBlock = textBlock
                    continue

                distancePreviousNext = distanceBetweenBlocks(image, textBlockNextToDescriptionBlock)
                if 0 < distanceToNextTextBlock < distancePreviousNext:
                    textBlockNextToDescriptionBlock = textBlock

        surroundingBlocks.remove(image)

        return textBlockNextToDescriptionBlock

    def identify_surrounding_text_blocks(self, data: PDF_Extraction) -> None:
        '''
        Identifies Textblocks which are around the Descriptions of images
        :return:
        '''

        def is_centered(textBlock, other_textBlocks):
            textBlocks = [_ for _ in other_textBlocks if _.isPartOfText]
            centers = [_.posX1 + (_.posX2 - _.posX1) / 2 for _ in textBlocks]
            center_tBlock = textBlock.posX1 + (textBlock.posX2 - textBlock.posX1) / 2
            for center in centers:
                if center * 0.9 <= center_tBlock <= center * 1.1:
                    return False
            return True

        for image in data.images:
            textBlocks = [_ for _ in data.textBlocks if _.pageNum == image.pageNum]
            textBlock = None

            for tBlock in textBlocks:
                if image.descriptionText == tBlock.text:
                    textBlock = tBlock

            textBlocks.remove(textBlock)

            is_in_center: bool = is_centered(textBlock, data.textBlocks)

            if is_in_center:
                image.set_upper_and_lower(textBlock, textBlocks)
            else:
                image.setSurroundingTextBlocks(textBlock, textBlocks)

    def identify_image_descriptions(self, data: PDF_Extraction) -> List[TextBlock]:
        '''
        Identifies Descriptions of the Images
        :return:
        '''
        res = []
        for textBlock in data.textBlocks:
            text = textBlock.text.lstrip().lower()
            if text.startswith("fig.") or text.startswith("figure"):
                # If the textBlock contains  more as 1000 Chars it is
                # probably not be an image_file description
                if len(text) < 1000:
                    res.append(textBlock)
        return res

    def _set_images(self) -> 'Image':
        print("Create Images")

    def _extract_images(self, data: PDF_Extraction) -> None:
        ''' Get the tables as images and saves them'''
        for image in data.grobid_results['images']:
            image.process_image(data.pages)

    def _delete_images_from_text(self, data: PDF_Extraction) -> None:
        ''' Deletes the tables form the text part. '''
        images = data.grobid_results['images']
        chapters = data.grobid_results['text']['chapters']

        # Deltes sentences from the images.
        for chapter in chapters:
            for image in images:
                for paragraph in chapter.paragraphs:
                    for chapter_sentence in paragraph.sentences:
                        if image.image_description.text == chapter_sentence.text:
                            paragraph.remove_sentence(chapter_sentence)

        # Deletes empty paragraphs
        for chapter in chapters:
            for paragraph in chapter.paragraphs:
                if len(paragraph.sentences) == 0:
                    chapter.remove_paragraph(paragraph)

        # Deletes empty chapters
        removable_chapters = [_ for _ in chapters if len(_.paragraphs) == 0]
        data.grobid_results['text']['chapters'] = [chapter for chapter in chapters if chapter not in removable_chapters]
