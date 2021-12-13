from __future__ import annotations
from typing import List, Type, Tuple, Dict
from core.extraction_modul.datamodels.internal_models import TextBlock
from segtok.segmenter import split_single
from core.detection_models.text_detection import is_grammatically_sentence
import core.schemas.datamodels as io
import re
import base64
import os
from core.config import TMP_DIRECTORY
import fitz


class Image:
    MAX_HEIGHT_OF_PAGE: int = 9999
    MIN_HEIGHT_OF_PAGE: int = 0
    MAX_WIDTH_OF_PAGE: int = 9999
    MIN_WIDTH_OF_PAGE: int = 0

    def __init__(self, datablock):
        self.pageNum: int = datablock.pageNum
        self.position: Tuple[int, int, int, int] = datablock.position
        self.posX1: int = datablock.position[0] if datablock.position[0] > 0 else 0
        self.posY1: int = datablock.position[1] if datablock.position[1] > 0 else 0
        self.posX2: int = datablock.position[2] if datablock.position[2] > 0 else 0
        self.posY2: int = datablock.position[3] if datablock.position[3] > 0 else 0
        self.descriptionText = datablock.text
        self.pathToImage: str = ""
        self.orientation = datablock.orientation
        self.surroundingDataBlocks: List[TextBlock] = []
        self.coordinatesOfPicture = [0, 0, 1, 1]
        self.is_image: bool = True
        self.textBlocksOfImage: List[TextBlock] = [datablock]
        self.possible_images = []
        self.image_file = None
        self.in_prediction = False
        datablock.isPartOfImage = True

    def get_image_of_image(self, pages):
        posX1 = self.coordinatesOfPicture[0]
        posY1 = self.coordinatesOfPicture[1]
        posX2 = self.coordinatesOfPicture[2]
        posY2 = self.coordinatesOfPicture[3]
        page = self.pageNum

        self.pathToImage = os.path.join(TMP_DIRECTORY, f"imgs/{page}_{posX1}_{posY1}_{posX2}_{posY2}.png")
        for site, _page in enumerate(pages):  # iterate through the pages
            if site == page:
                try:
                    mat = fitz.Matrix(1, 1)  # zoom factor 2 in each direction
                    clip = fitz.IRect(posX1, posY1, posX2, posY2)
                    pix = _page.getPixmap(matrix=mat, clip=clip)
                    pix.writePNG(self.pathToImage)
                    self.image_file = open(self.pathToImage, "rb").read()
                except RuntimeError:
                    clip = fitz.IRect(posX1, posY1, posX2, posY2)
                    pix = _page.getPixmap(irect=clip)
                    pix.writePNG(self.pathToImage)
                    self.image_file = open(self.pathToImage, "rb").read()
                break

    def get_image_name(self) -> str:
        image_name: str = ""
        text = " ".join([_.text for _ in self.textBlocksOfImage])
        match = re.search("^fig(\.|ure) +\d+", text.lower())
        if match:
            image_name = text[match.start(): match.end()]
        return image_name

    def to_io(self) -> io.Image:
        image_name: str = self.get_image_name()

        encoded_img_string = base64.urlsafe_b64encode(self.image_file).decode('utf-8')
        return io.Image(**{
            "base64_file": encoded_img_string,
            "description": " ".join([_.text for _ in self.textBlocksOfImage]),
            "name": image_name
        })

    def save_as_dict(self):
        with open(self.pathToImage, "rb") as image_file:
            encoded_img_string = base64.urlsafe_b64encode(image_file.read()).decode('utf-8')
        return {
            'image_base64': encoded_img_string,
            'description': self.descriptionText
        }

    def is_inside(self, tBlock: TextBlock):
        posX1, posY1, posX2, posY2 = self.coordinatesOfPicture

        if tBlock.pageNum == self.pageNum:
            if posY1 < tBlock.posY1 < posY2 or posY1 < tBlock.posY2 < posY2:
                if posX1 < tBlock.posX1 < posX2 or posX1 < tBlock.posX2 < posX2:
                    return True
        return False

    @classmethod
    def setPageCoordinates(cls, max_height, min_height, max_width, min_width):
        Image.MAX_HEIGHT_OF_PAGE = max_height
        Image.MIN_HEIGHT_OF_PAGE = min_height
        Image.MAX_WIDTH_OF_PAGE = max_width
        Image.MIN_WIDTH_OF_PAGE = min_width

    def checkIsImage(self, images) -> None:
        # If the Text contains more as two sentences it is probably not a ImageDescription
        #has_some_sentences: bool =  self.hasNumberOfSentences() > 2
        is_between_two_sentences: bool = self.is_between_two_sentences()
        check_for_multiple_existence: bool = self.check_multiple_existence(images)
        is_in_image_prediction: bool = self.in_prediction
        if (is_between_two_sentences or check_for_multiple_existence) and not is_in_image_prediction:
            self.is_image = False

    def check_multiple_existence(self, images: List[Image]) -> bool:
        image_name = re.search("\d+", self.descriptionText) # First occurence of an number
        for image in images:
            if image is self:
                continue
            other_name = re.search("\d+", image.descriptionText)
            # If the Same image name comes multiple times calculate the distance to the image and
            # check if the distance is shorter than the other one.
            if image_name.group() == other_name.group():
                area_other = ((image.posX2 - image.posX1) * (image.posY2 - image.posY1)) ** 2
                area = ((self.posX2 - self.posX1) * (self.posY2 - self.posY1)) ** 2
                if area > area_other:
                    return True
        return False


    def is_between_two_sentences(self) -> bool:
        res = False
        self.surroundingDataBlocks.sort(key=lambda x: x.absY1)
        textBlocks_above = []
        textBlocks_below = []

        x1 = min([_.posX1 for _ in self.textBlocksOfImage])
        y1 = min([_.posY1 for _ in self.textBlocksOfImage])
        x2 = max([_.posX2 for _ in self.textBlocksOfImage])
        y2 = max([_.posY2 for _ in self.textBlocksOfImage])


        for tBlock in self.surroundingDataBlocks:
            if x1 <= (tBlock.posX2 - tBlock.posX1)/2 + tBlock.posX1 <= x2:
                if not tBlock.isRecurringElement and tBlock.isPartOfText and not tBlock.isPartOfTable and not tBlock.isPartOfMeta:
                    if tBlock.posY2 < y1:
                        textBlocks_above.append(tBlock)
                    elif tBlock.posY1 > y2:
                        textBlocks_below.append(tBlock)


        if len(textBlocks_above) == 0 or len(textBlocks_below) == 0:
            return False

        if (0 <= self.posY1 - textBlocks_above[-1].posY1 <= TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES) \
                and (0 <= textBlocks_below[0].posY2 - self.posY2 <= TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES):
            res = True
        return res
    def hasNumberOfSentences(self):
        sentenceCounter = 0
        sentences = split_single(self.descriptionText)
        for sentence in sentences:
            if is_grammatically_sentence(sentence):
                sentenceCounter += 1
        return sentenceCounter

    def set_upper_and_lower(self, imageBlock, textBlocks):
        for tBlock in textBlocks:
            if imageBlock.posY1 > tBlock.posY2 or imageBlock.posY2 < tBlock.posY1:
                if tBlock.isPartOfText or tBlock.isRecurringElement or tBlock.isPartOfImage or tBlock.isPartOfTable:
                    self.surroundingDataBlocks.append(tBlock)

    def setSurroundingTextBlocks(self, imageBlock, textBlocks: List[TextBlock]):
        for tBlock in textBlocks:
            is_above_or_below: bool = imageBlock.posY1 > tBlock.posY2 or imageBlock.posY2 < tBlock.posY1
            is_left_or_right: bool = imageBlock.posX1 > tBlock.posX2 or imageBlock.posX2 < tBlock.posX1
            if is_above_or_below or is_left_or_right:
                if tBlock.isPartOfText or tBlock.isRecurringElement or tBlock.isPartOfImage or tBlock.isPartOfTable:
                    self.surroundingDataBlocks.append(tBlock)



    def setCoordinatesOfPicture(self):
        '''
        A description of the idea of this algorithmn can be found in the documentation.
        :return:
        '''

        # TBlockR := Textblock Right To Description Block
        TBlockL: TextBlock = self.getTextNextToDescriptionBlock(self.surroundingDataBlocks, 0)  # Evtl hier
        TBlockR: TextBlock = self.getTextNextToDescriptionBlock(self.surroundingDataBlocks, 1)
        TBlockO: TextBlock = self.getTextNextToDescriptionBlock(self.surroundingDataBlocks, 2)
        TBlockU: TextBlock = self.getTextNextToDescriptionBlock(self.surroundingDataBlocks, 3)

        # pictureIsBellow: bool = imageExtractor.HEIGHT_OF_PAGE - self.posY2 > self.posY1 - TBlockO.posY2

        posX1 = TBlockL.posX2 if TBlockL is not None else Image.MIN_WIDTH_OF_PAGE
        posX2 = TBlockR.posX1 if TBlockR is not None else Image.MAX_WIDTH_OF_PAGE
        posY1 = TBlockO.posY2 if TBlockO is not None else Image.MIN_HEIGHT_OF_PAGE
        posY2 = TBlockU.posY1 if TBlockU is not None else Image.MAX_HEIGHT_OF_PAGE
        self.calculated_boundaries = [posX1, posY1, posX2, posY2]

    def _getFirstCoordinate(self, textBlock, side):
        # To the right
        if side == 0:
            return textBlock.posX1
        # To the left
        elif side == 1:
            return -textBlock.posX2
        # To Above
        elif side == 2:
            return textBlock.posY1
        # To Below
        elif side == 3:
            return -textBlock.posY2

    def _getSecondCoordinate(self, textBlock, side):
        if side == 0:
            return textBlock.posX2
        elif side == 1:
            return -textBlock.posX1
        elif side == 2:
            return textBlock.posY2
        elif side == 3:
            return -textBlock.posY1

    def _getOnlyRelevantTextBlocks(self, surroundingBlocks, side):

        def check_text_conditions(textBlock: TextBlock) -> bool:
            is_longer_as_a_few_words: bool = len(textBlock.text.split(" ")) > 2
            end_with_point: bool = re.search("\.$", textBlock.text.rstrip())
            return is_longer_as_a_few_words and not end_with_point

        res = []

        for textBlock in surroundingBlocks:

            if isinstance(textBlock, TextBlock):
                # if textBlock.isPartOfMeta:
                #    res.append(textBlock)
                #    continue
                is_part_of_text: bool = check_text_conditions(textBlock)
                if not is_part_of_text:
                    continue
            if side == 0 or side == 1:
                if self.posY1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= textBlock.posY1 <= self.posY2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES or self.posY1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= textBlock.posY2 <= self.posY2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES:
                    res.append(textBlock)
                elif textBlock.posY1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= self.posY1 <= textBlock.posY2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES or textBlock.posY1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= self.posY2 <= textBlock.posY2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES:
                    res.append(textBlock)
            else:

                if self.posX1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= textBlock.posX1 <= self.posX2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES or self.posX1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= textBlock.posX2 <= self.posX2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES:
                    res.append(textBlock)
                elif textBlock.posX1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= self.posX1 <= textBlock.posX2 + 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES or textBlock.posX1 - 10 * TextBlock.NORMAL_DISTANCE_BETWEEN_TWO_LINES <= self.posX2 <= textBlock.posX2:
                    res.append(textBlock)

        return res

    def getTextNextToDescriptionBlock(self, surroundingBlocks, side: int = 0):
        surroundingBlocks.append(self)

        distanceBetweenBlocks = lambda block1, block2: self._getFirstCoordinate(block1,
                                                                                side) - self._getSecondCoordinate(
            block2, side)
        focusedBlocks: List[TextBlock] = self._getOnlyRelevantTextBlocks(surroundingBlocks, side)
        textBlockNextToDescriptionBlock = None
        for textBlock in focusedBlocks:

            distanceToNextTextBlock = distanceBetweenBlocks(self, textBlock)
            if 0 < distanceToNextTextBlock:
                if textBlockNextToDescriptionBlock is None:
                    textBlockNextToDescriptionBlock = textBlock
                    continue

                distancePreviousNext = distanceBetweenBlocks(self, textBlockNextToDescriptionBlock)
                if 0 < distanceToNextTextBlock < distancePreviousNext:
                    textBlockNextToDescriptionBlock = textBlock

        surroundingBlocks.remove(self)

        return textBlockNextToDescriptionBlock
