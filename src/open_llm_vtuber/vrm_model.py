import json
import chardet
from loguru import logger

# VrmModel provides the same public interface as Live2dModel so it can be used
# as a drop-in replacement throughout the codebase (duck typing).
# Key difference: emotionMap values are VRM BlendShape preset name strings
# (e.g. "happy", "angry") instead of integer expression indices.


class VrmModel:
    """
    A class to represent a VRM 3D model. Provides the same public interface as
    Live2dModel so it integrates transparently with the existing pipeline.

    Attributes:
        model_dict_path (str): The path to the model dictionary file.
        live2d_model_name (str): The name of the VRM model (reuses field name for compat).
        model_info (dict): The information of the VRM model from model_dict.json.
        emo_map (dict): Mapping of emotion keywords to VRM BlendShape preset names.
        emo_str (str): Comma-separated emotion keywords for injection into LLM prompt.
    """

    model_dict_path: str
    live2d_model_name: str
    model_info: dict
    emo_map: dict
    emo_str: str

    def __init__(
        self, live2d_model_name: str, model_dict_path: str = "model_dict.json"
    ):
        self.model_dict_path: str = model_dict_path
        self.live2d_model_name: str = live2d_model_name
        self.set_model(live2d_model_name)

    def set_model(self, model_name: str) -> None:
        """
        Set the model with its name and load the model information.

        Parameters:
            model_name (str): The name of the VRM model.
        """
        self.model_info: dict = self._lookup_model_info(model_name)
        self.emo_map: dict = {
            k.lower(): v for k, v in self.model_info["emotionMap"].items()
        }
        self.emo_str: str = " ".join([f"[{key}]," for key in self.emo_map.keys()])

    def _load_file_content(self, file_path: str) -> str:
        """Load the content of a file with robust encoding handling."""
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "ascii"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue

        try:
            with open(file_path, "rb") as file:
                raw_data = file.read()
            detected = chardet.detect(raw_data)
            detected_encoding = detected["encoding"]

            if detected_encoding:
                try:
                    return raw_data.decode(detected_encoding)
                except UnicodeDecodeError:
                    pass
        except Exception as e:
            logger.error(f"Error detecting encoding for {file_path}: {e}")

        raise UnicodeError(f"Failed to decode {file_path} with any encoding")

    def _lookup_model_info(self, model_name: str) -> dict:
        """
        Find the model information from the model dictionary.

        Parameters:
            model_name (str): The name of the VRM model.

        Returns:
            dict: The dictionary with the information of the matched model.
        """
        self.live2d_model_name = model_name

        try:
            file_content = self._load_file_content(self.model_dict_path)
            model_dict = json.loads(file_content)
        except FileNotFoundError as file_e:
            logger.critical(
                f"Model dictionary file not found at {self.model_dict_path}."
            )
            raise file_e
        except json.JSONDecodeError as json_e:
            logger.critical(
                f"Error decoding JSON from model dictionary file at {self.model_dict_path}."
            )
            raise json_e
        except UnicodeError as uni_e:
            logger.critical(
                f"Error reading model dictionary file at {self.model_dict_path}."
            )
            raise uni_e
        except Exception as e:
            logger.critical(
                f"Error occurred while reading model dictionary file at {self.model_dict_path}."
            )
            raise e

        matched_model = next(
            (model for model in model_dict if model["name"] == model_name), None
        )

        if matched_model is None:
            logger.critical(f"Unable to find {model_name} in {self.model_dict_path}.")
            raise KeyError(
                f"{model_name} not found in model dictionary {self.model_dict_path}."
            )

        logger.info("VRM Model Information Loaded.")
        return matched_model

    def extract_emotion(self, str_to_check: str) -> list:
        """
        Check the input string for emotion keywords and return a list of
        VRM BlendShape preset name strings (e.g. ["happy", "angry"]).

        Parameters:
            str_to_check (str): The string to check for emotions.

        Returns:
            list: A list of VRM BlendShape preset name strings.
                  Empty list if no emotions found.
        """
        expression_list = []
        str_to_check_lower = str_to_check.lower()

        i = 0
        while i < len(str_to_check_lower):
            if str_to_check_lower[i] != "[":
                i += 1
                continue
            for key in self.emo_map.keys():
                emo_tag = f"[{key}]"
                if str_to_check_lower[i : i + len(emo_tag)] == emo_tag:
                    expression_list.append(self.emo_map[key])
                    i += len(emo_tag) - 1
                    break
            i += 1
        return expression_list

    def remove_emotion_keywords(self, target_str: str) -> str:
        """
        Remove the emotion keywords from the input string.

        Parameters:
            target_str (str): The string to clean.

        Returns:
            str: The cleaned string with the emotion keywords removed.
        """
        lower_str = target_str.lower()

        for key in self.emo_map.keys():
            lower_key = f"[{key}]".lower()
            while lower_key in lower_str:
                start_index = lower_str.find(lower_key)
                end_index = start_index + len(lower_key)
                target_str = target_str[:start_index] + target_str[end_index:]
                lower_str = lower_str[:start_index] + lower_str[end_index:]
        return target_str
