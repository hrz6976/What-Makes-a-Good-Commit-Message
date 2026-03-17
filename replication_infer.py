import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel, BertTokenizer

try:
    import allennlp_models.structured_prediction  # noqa: F401
    from allennlp.predictors.predictor import Predictor
except ImportError:
    Predictor = None


REPO_ROOT = Path(__file__).resolve().parent
WHY_MODEL_PATH = Path(os.environ.get("WHY_MODEL_PATH", REPO_ROOT / "outputs" / "why_bert_bilstm_reproduced.pth"))
WHAT_MODEL_PATH = Path(os.environ.get("WHAT_MODEL_PATH", REPO_ROOT / "outputs" / "what_bert_bilstm_reproduced.pth"))
ALLENNLP_MODEL_DIR = Path(os.environ.get("ALLENNLP_MODEL_DIR", REPO_ROOT / "Model"))
BERT_NAME = "bert-base-uncased"
MAX_LENGTH = 200


def normalize_message(text: str) -> str:
    return text.replace("\n", " <enter> ").replace("\t", " <tab> ").strip()


def get_allennlp_predictor():
    global _ALLENNLP_PREDICTOR
    if _ALLENNLP_PREDICTOR is not None:
        return _ALLENNLP_PREDICTOR
    if Predictor is None or not ALLENNLP_MODEL_DIR.exists():
        return None
    try:
        config = json.loads((ALLENNLP_MODEL_DIR / "config.json").read_text())
        elmo = config["model"]["text_field_embedder"].pop("elmo")
        elmo["options_file"] = str(ALLENNLP_MODEL_DIR / "fta" / "model.text_field_embedder.elmo.options_file")
        elmo["weight_file"] = str(ALLENNLP_MODEL_DIR / "fta" / "model.text_field_embedder.elmo.weight_file")
        config["model"]["text_field_embedder"] = {"token_embedders": {"elmo": elmo}}
        config["model"]["evalb_directory_path"] = None
        _ALLENNLP_PREDICTOR = Predictor.from_path(
            str(ALLENNLP_MODEL_DIR),
            predictor_name="constituency_parser",
            overrides=json.dumps(config),
        )
    except Exception:
        _ALLENNLP_PREDICTOR = None
    return _ALLENNLP_PREDICTOR


def replace_identifiers_with_allennlp(message: str) -> str:
    predictor = get_allennlp_predictor()
    if predictor is None:
        return replace_identifiers(message)
    try:
        tokens, tags, length = allennlp_tag_like(message, predictor)
        indices, token_list = filter_candidate_tokens(length, tokens, tags)
        if not indices:
            return replace_identifiers(message)
        found_tokens = []
        for index in indices:
            token = token_list[index]
            if index > 0 and index < len(token_list) - 1 and token_list[index - 1] == "'" and token_list[index + 1] == "'":
                found_tokens.append("'" + str(token) + "'")
            else:
                found_tokens.append(token)
        if not found_tokens:
            return replace_identifiers(message)
        return replace_tokens_in_message(message, sorted(set(found_tokens), key=len, reverse=True))
    except Exception:
        return replace_identifiers(message)


def allennlp_tag_like(message: str, predictor):
    result = predictor.predict(message)
    tokens = result["tokens"]
    tags = result["pos_tags"]

    indices = []
    for i, token in enumerate(tokens):
        s = str(token)
        if s.startswith(("file_name>", "version>", "url>", "enter>", "tab>", "iden>", "method_name>", "pr_link>", "issue_link>", "otherCommit_link>")):
            indices.append(i)
        elif s.endswith(("<file_name", "<version", "<url", "<enter", "<tab", "<iden", "<method_name", "<pr_link", "<issue_link", "<otherCommit_link")):
            indices.append(i)

    new_tokens = []
    new_tags = []
    for i, token in enumerate(tokens):
        if i in indices:
            s = str(token)
            prefixes = [
                "file_name", "method_name", "version", "url", "enter", "tab",
                "iden", "pr_link", "issue_link", "otherCommit_link",
            ]
            handled = False
            for prefix in prefixes:
                if s.startswith(prefix + ">"):
                    s = s.replace(prefix + ">", "")
                    new_tokens.extend([prefix, ">", s])
                    new_tags.extend(["XX", "XX", "XX"])
                    handled = True
                    break
                if s.endswith("<" + prefix):
                    s = s.replace("<" + prefix, "")
                    new_tokens.extend([s, "<", prefix])
                    new_tags.extend(["XX", "XX", "XX"])
                    handled = True
                    break
            if not handled:
                new_tokens.append(tokens[i])
                new_tags.append(tags[i])
        else:
            new_tokens.append(tokens[i])
            new_tags.append(tags[i])

    tokens = new_tokens
    tags = new_tags
    merged_tokens = []
    merged_tags = []
    targets = ["file_name", "version", "url", "enter", "tab", "iden", "issue_link", "pr_link", "otherCommit_link", "method_name"]
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 2 and tokens[i] == "<" and tokens[i + 1] in targets and tokens[i + 2] == ">":
            merged_tokens.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
            merged_tags.append("XX")
            i += 3
        else:
            merged_tokens.append(tokens[i])
            merged_tags.append(tags[i])
            i += 1

    return merged_tokens, merged_tags, len(merged_tokens)


def filter_candidate_tokens(length, tokens, tags):
    indices = []
    for i in range(1, length):
        token = str(tokens[i])
        tag = str(tags[i])
        if token.startswith("@"):
            indices.append(i)
        elif token.isalnum() and not token.islower():
            if tag.startswith("NN"):
                indices.append(i)
            else:
                before = i > 0 and str(tokens[i - 1]) == "'"
                after = i + 1 < len(tokens) and str(tokens[i + 1]) == "'"
                if before and after:
                    indices.append(i)
    return indices, tokens


def get_unreplacable_indices(message: str, replacement: str) -> list[int]:
    unreplacable_indices = []
    start = 0
    index = str(message).find(replacement, start, len(message))
    while index > -1:
        start = index + len(replacement)
        for i in range(index, start):
            unreplacable_indices.append(i)
        index = str(message).find(replacement, start, len(message))
    return unreplacable_indices


def replace_tokens_in_message(message: str, tokens: list[str]) -> str:
    unreplacable = []
    replacements = [
        "<file_name>", "<version>", "<url>", "<enter>", "<tab>",
        "<issue_link>", "<pr_link>", "<otherCommit_link>", "<method_name>",
    ]
    for replacement in replacements:
        unreplacable += get_unreplacable_indices(message, replacement)

    locations = []
    for token in tokens:
        end = 0
        while end < len(message):
            start = str(message).find(token, end, len(message))
            if start == -1:
                break
            end = start + len(token)
            before = start > 0 and str(message[start - 1]).isalnum()
            after = end < len(message) and str(message[end]).isalnum()
            overlaps_placeholder = any(index in unreplacable for index in range(start, end))
            if not before and not after and not overlaps_placeholder:
                locations.append([start, end])

    locations.sort(key=lambda item: item[0])
    i = 0
    while i < len(locations) - 1:
        if locations[i][1] > locations[i + 1][0]:
            if locations[i][0] == locations[i + 1][0]:
                if locations[i][1] < locations[i + 1][1]:
                    locations.pop(i)
                elif locations[i][1] > locations[i + 1][1]:
                    locations.pop(i + 1)
                else:
                    locations.pop(i + 1)
            elif locations[i][0] < locations[i + 1][0] and locations[i][1] >= locations[i + 1][1]:
                locations.pop(i + 1)
            else:
                i += 1
        else:
            i += 1

    merged_locations = []
    i = 0
    start = -1
    while i < len(locations):
        if start < 0:
            start = locations[i][0]
        if i < len(locations) - 1 and locations[i + 1][0] - locations[i][1] < 2:
            i += 1
            continue
        end = locations[i][1]
        merged_locations.append([start, end])
        start = -1
        i += 1

    end = 0
    new_message = ""
    for location in merged_locations:
        start = location[0]
        new_message += message[end:start]
        new_message += "<iden>"
        end = location[1]
    new_message += message[end:len(message)]
    return new_message


def load_messages(input_file: Path, text_field: str) -> list[dict]:
    if input_file.suffix == ".jsonl":
        records = []
        for line in input_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            records.append({"message": data[text_field], "source": data})
        return records

    if input_file.suffix == ".csv":
        with input_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [{"message": row[text_field], "source": row} for row in reader]

    return [
        {"message": line.strip(), "source": {"message": line.strip()}}
        for line in input_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class BertBiLSTM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        output_size: int = 2,
        n_layers: int = 2,
        bidirectional: bool = True,
        drop_prob: float = 0.55,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        config = BertConfig.from_pretrained(BERT_NAME)
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(
            768,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_size)

    def forward(self, x: torch.Tensor, hidden):
        attention_mask = (x != 0).long()
        x = self.bert(input_ids=x, attention_mask=attention_mask)[0]
        _, (hidden_last, _) = self.lstm(x, hidden)
        if self.bidirectional:
            hidden_last_out = torch.cat([hidden_last[-2], hidden_last[-1]], dim=-1)
        else:
            hidden_last_out = hidden_last[-1]
        out = self.dropout(hidden_last_out)
        return self.fc(out)

    def init_hidden(self, batch_size: int, device: torch.device):
        number = 2 if self.bidirectional else 1
        shape = (self.n_layers * number, batch_size, self.hidden_dim)
        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


class CommitMessageClassifier:
    def __init__(self, device: Optional[str] = None) -> None:
        self.device = resolve_device(device)
        self.tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
        self.why_model = self._load_model(WHY_MODEL_PATH)
        self.what_model = self._load_model(WHAT_MODEL_PATH)

    def _load_model(self, checkpoint_path: Path) -> BertBiLSTM:
        model = BertBiLSTM().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def _predict_scores(self, model: BertBiLSTM, messages: list[str]) -> list[float]:
        encoded = self.tokenizer(
            [normalize_message(message) for message in messages],
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        hidden = model.init_hidden(batch_size=len(messages), device=self.device)
        with torch.no_grad():
            logits = model(input_ids, hidden)
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.cpu().tolist()

    def predict_batch(self, messages: list[str], batch_size: int = 32) -> list[dict]:
        outputs = []
        for start in range(0, len(messages), batch_size):
            batch = messages[start:start + batch_size]
            why_scores = self._predict_scores(self.why_model, batch)
            what_scores = self._predict_scores(self.what_model, batch)
            for message, why_score, what_score in zip(batch, why_scores, what_scores):
                contains_why = why_score >= 0.5
                contains_what = what_score >= 0.5
                outputs.append(
                    {
                        "message": message,
                        "contains_why": contains_why,
                        "contains_what": contains_what,
                        "is_good": contains_why and contains_what,
                        "why_score": round(float(why_score), 6),
                        "what_score": round(float(what_score), 6),
                        "device": str(self.device),
                    }
                )
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether commit messages contain Why and What.")
    parser.add_argument("message", nargs="?", help="Single commit message. If omitted, use --input-file or stdin.")
    parser.add_argument("--input-file", type=Path, help="Path to .txt, .jsonl, or .csv file.")
    parser.add_argument("--output-file", type=Path, help="Optional output file for batch mode. Defaults to stdout.")
    parser.add_argument("--text-field", default="message", help="Field name for .jsonl or .csv batch inputs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size for batch mode.")
    parser.add_argument("--device", help="Force torch device, e.g. cpu or cuda")
    args = parser.parse_args()

    predictor = CommitMessageClassifier(device=args.device)

    if args.input_file is not None:
        records = load_messages(args.input_file, args.text_field)
        predictions = predictor.predict_batch([record["message"] for record in records], batch_size=args.batch_size)
        lines = []
        for record, prediction in zip(records, predictions):
            merged = dict(record["source"])
            merged.update(prediction)
            lines.append(json.dumps(merged, ensure_ascii=False))
        output = "\n".join(lines)
        if args.output_file is not None:
            args.output_file.write_text(output + "\n", encoding="utf-8")
        else:
            print(output)
        return

    message = args.message if args.message is not None else input().strip()
    print(json.dumps(predictor.predict_batch([message], batch_size=1)[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
