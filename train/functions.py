import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from model import SRL_MODEL
from torch.nn.utils import clip_grad_norm_


def smooth_labels(labels: torch.Tensor, epsilon: float = 0.1):
    """
    Smooth the labels using label smoothing

    Parameters:
        labels: The labels
        epsilon: The epsilon for the label smoothing

    Returns:
        The smoothed labels
    """
    with torch.no_grad():
        # add random noise to the labels
        random_noise = torch.rand(labels.shape).to(labels.device)
        sum_noise = random_noise.sum(dim=-1, keepdim=True)
        random_noise = (
            random_noise / sum_noise * (epsilon * labels.sum(dim=-1, keepdim=True))
        )
        labels = labels * (1 - epsilon) + random_noise

    return labels


# RELATION FUNCTIONS


def relation_loss(
    mask: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, noise: float = 0.2
):
    """
    Compute loss for relational classification

    Parameters:
        mask: The mask for the labels
        logits: The logits from the model
        labels: The labels for the relations
        noise: The noise to add to the labels

    Returns:
        The loss, accuracy, precision, recall, and f1 score for the relation classification
    """
    # Compute loss for relational classification
    logits = logits * mask

    # Calculate positive weight for relational classification
    pos_weight_rel = ((labels == 0).float() * mask).sum() / (labels == 1).float().sum()

    # Use BCEWithLogitsLoss with pos_weight
    loss_function_relation_weighted = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight_rel.to(logits.device), reduction="mean"
    )
    # label smoothing
    labels_with_noise = smooth_labels(labels.float(), epsilon=noise)
    relational_loss = loss_function_relation_weighted(logits, labels_with_noise)

    # Compute accuracy and F1 score for relational classification
    with torch.no_grad():
        rel_preds = (torch.sigmoid(logits) > 0.5).float().flatten()
        rel_labels = labels.float().flatten()
        rel_accuracy = (rel_preds == rel_labels).float().mean()

        rel_labels = rel_labels.cpu().numpy()
        rel_preds = rel_preds.cpu().numpy()
        rel_precision, rel_recall, rel_f1, _ = precision_recall_fscore_support(
            rel_labels, rel_preds, average="binary", zero_division=1
        )

    return relational_loss, rel_accuracy, rel_precision, rel_recall, rel_f1


# SENSES FUNCTIONS


def senses_loss(logits: torch.Tensor, labels: torch.Tensor, noise: float = 0.2):
    """
    Compute loss for sense classification

    Parameters:
        logits: The logits from the model
        labels: The labels for the senses
        noise: The noise to add to the labels

    Returns:
        The loss, accuracy, precision, recall, and f1 score for the sense classification
    """
    # Compute loss for sense classification

    criterion = nn.CrossEntropyLoss(reduction="mean")
    # label smoothing
    labels_with_noise = smooth_labels(labels.float(), epsilon=noise)
    loss = criterion(logits, labels_with_noise)
    # loss /= logits.size(0)

    # Compute accuracy and F1 score for sense classification
    with torch.no_grad():
        sense_preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        sense_acc = (sense_preds == labels).float().mean()
        sense_preds = sense_preds.cpu().numpy()
        labels = labels.cpu().numpy()

        sense_precision, sense_recall, sense_f1, _ = precision_recall_fscore_support(
            labels, sense_preds, average="weighted", zero_division=1
        )

    return loss, sense_acc, sense_precision, sense_recall, sense_f1


# ROLES FUNCTIONS


def get_spans(role_logits, threshold=0.5):
    """
    Get the spans of the roles from the logits

    Parameters:
        role_logits: The logits of the roles
        threshold: The threshold for the probability
        mode: The mode to use for the spans (t for taking the most probable, a for aggregating all)

    Returns:
        The spans of the roles
    """
    role_spans = {}

    for idx, logits in enumerate(role_logits):
        for role_idx, logit in enumerate(logits):
            prob = torch.sigmoid(logit)
            if prob > threshold:
                if role_idx not in role_spans:
                    role_spans[role_idx] = [([idx], [prob])]
                else:
                    if (
                        role_spans[role_idx][-1][0][-1] + 1
                    ) == idx:  # Check if the current token is contiguous with the previous one
                        role_spans[role_idx][-1][0].append(idx)
                        role_spans[role_idx][-1][1].append(prob)
                    else:  # If not, start a new span
                        role_spans[role_idx].append(([idx], [prob]))

    final_spans = {}
    for role_idx, spans in role_spans.items():
        # take the most probable (use mean)
        max_prob = 0
        max_span = None
        for span in spans:
            prob = torch.mean(torch.tensor(span[1]))
            if prob > max_prob:
                max_prob = prob
                max_span = span
        final_spans[role_idx] = max_span[0]

    return final_spans


def top_span(logits, threshold=0.5):
    """
    Select the top span for each role

    Parameters:
        logits: The logits from the model

    Returns:
        The masked logits to have only the top span for each role
    """
    out = torch.zeros_like(logits)

    for rel in range(logits.shape[0]):
        spans = get_spans(logits[rel], threshold=threshold)

        for role in spans:
            out[rel, spans[role], role] = 1

    return out


def role_loss(
    results: list[torch.Tensor],
    labels: list[torch.Tensor],
    top: bool = False,
    group: bool = False,
    threshold: float = 0.5,
    weight_role_labels: bool = False,
    noise: float = 0.2,
):
    """
    Compute loss for role classification

    Parameters:
        results: The logits from the model
        labels: The labels for the roles
        top: Wether to use top selection for the predicted spans
        group: Wether to group the roles to compute the loss
        threshold: The threshold for the role classification
        weight_role_labels: Wether to weight the role labels in the loss function
        noise: The noise to add to the labels

    Returns:
        The loss, accuracy, precision, recall, and f1 score for the role classification
    """
    # Compute loss for role classification
    role_loss = 0
    all_role_preds = []
    all_role_labels = []
    correct_roles = 0
    total_roles = 0

    roles_logits = torch.tensor([]).to(results[0].device)
    roles_labels = torch.tensor([]).to(results[0].device)

    for i in range(len(results)):
        role_logits = results[i]

        role_labels = labels[i].to(role_logits.device).float()

        if group:
            roles_logits = torch.cat(
                (roles_logits, role_logits.view(-1, role_logits.shape[-1])), dim=0
            )
            roles_labels = torch.cat(
                (roles_labels, role_labels.view(-1, role_labels.shape[-1])), dim=0
            )
        else:
            with torch.no_grad():
                # Calculate positive weight for role classification for each role
                if weight_role_labels:
                    pos_weight = torch.tensor(
                        [
                            max(1, (role_labels[:, :, i] == 0).float().sum().item())
                            / max(
                                (role_labels[:, :, i] == 0).float().sum().item() / 100,
                                (role_labels[:, :, i] == 1).float().sum().item(),
                                1,
                            )
                            for i in range(role_labels.shape[2])
                        ]
                    ).to(role_logits.device)
                else:
                    # Have all the roles to have the same weight for the loss
                    pos_weight = torch.ones(role_labels.shape[2]).to(role_logits.device)

                # Weighted Binary Cross-Entropy Loss
                criterion = torch.nn.BCEWithLogitsLoss(
                    pos_weight=pos_weight, reduction="none"
                )
                # label smoothing
                role_labels_with_noise = smooth_labels(role_labels, epsilon=noise)
            loss = (
                criterion(role_logits, role_labels_with_noise)
                .view(-1, role_labels.shape[-1])
                .mean(0)
                .sum()
            )
            role_loss += loss

        # Compute accuracy and F1 score for role classification
        if top:
            role_preds = top_span(role_logits, threshold=threshold)
        else:
            role_preds = (torch.sigmoid(role_logits) > threshold).float()

        for j in range(role_labels.shape[0]):  # for each relation
            for k in range(role_labels.shape[1]):  # for each word
                all_role_labels.append(role_labels[j][k].cpu().numpy())
                all_role_preds.append(role_preds[j][k].cpu().numpy())

    # compute accuracy for each role
    for i in range(len(all_role_labels)):
        for j in range(len(all_role_labels[i])):
            if all_role_preds[i][j] == all_role_labels[i][j]:
                correct_roles += 1
            total_roles += 1
    role_accuracy = correct_roles / total_roles

    # compute precision, recall and f1 for each role
    role_precision, role_recall, role_f1, _ = precision_recall_fscore_support(
        all_role_labels, all_role_preds, zero_division=1, average="micro"
    )

    # compute loss for each role
    if group:
        with torch.no_grad():
            if weight_role_labels:
                pos_weight = torch.tensor(
                    [
                        max(1, (roles_labels[:, i] == 0).float().sum().item())
                        / max(
                            (roles_labels[:, i] == 0).float().sum().item() / 100,
                            (roles_labels[:, i] == 1).float().sum().item(),
                            1,
                        )
                        for i in range(roles_labels.shape[1])
                    ]
                ).to(roles_logits.device)
            else:
                pos_weight = torch.ones(roles_labels.shape[1]).to(roles_logits.device)
            criterion = torch.nn.BCEWithLogitsLoss(
                pos_weight=pos_weight, reduction="none"
            )
            role_labels_with_noise = smooth_labels(roles_labels, epsilon=noise)
        role_loss = (
            criterion(roles_logits, role_labels_with_noise)
            .view(-1, roles_labels.shape[-1])
            .mean(0)
            .sum()
        )

    # role_loss /= len(results)

    return role_loss, role_accuracy, role_precision, role_recall, role_f1


# CUMULATIVE LOSS
def loss(
    model: SRL_MODEL,
    rel_mask: torch.Tensor,
    rel_logits: torch.Tensor,
    rel_labels: torch.Tensor,
    sense_logits: torch.Tensor,
    sense_labels: torch.Tensor,
    role_logits: torch.Tensor,
    role_labels: torch.Tensor,
    group_roles: bool = False,
    role_threshold: float = 0.5,
    weight_role_labels: bool = False,
    weight_rel: float = 1,
    weight_sense: float = 1,
    weight_role: float = 1,
    F1_loss_power: float = 2,
    noise: float = 0.2,
    top: bool = False,
):
    """
    Compute the total loss for the model

    Parameters:
        model: The model
        rel_mask: The mask for the relation labels
        rel_logits: The logits for the relation classification
        rel_labels: The labels for the relation classification
        sense_logits: The logits for the sense classification (not used)
        sense_labels: The labels for the sense classification (not used)
        role_logits: The logits for the role classification
        role_labels: The labels for the role classification
        group_roles: Wether to group the roles to compute the loss
        role_threshold: The threshold for the role classification
        weight_role_labels: Wether to weight the role labels in the loss function
        weight_rel: The weight for the relation classification
        weight_sense: The weight for the sense classification (not used)
        weight_role: The weight for the role classification
        F1_loss_power: The power for the rescaling using the F1 score
        noise: The noise to add to the labels
        top: Wether to use top selection for the predicted spans

    Returns:
        A dictionary containing the total loss, relation loss, sense loss, role loss, relation accuracy, relation precision, relation recall, relation f1, sense accuracy, sense precision, sense recall, sense f1, role accuracy, role precision, role recall, and role f1
    """
    # Compute loss for each task
    rel_loss, rel_accuracy, rel_precision, rel_recall, rel_f1 = relation_loss(
        rel_mask, rel_logits, rel_labels, noise
    )
    # sense_loss, sense_acc, sense_precision, sense_recall, sense_f1 = senses_loss(sense_logits, sense_labels, noise)
    sense_loss, sense_acc, sense_precision, sense_recall, sense_f1 = (
        torch.tensor(0),
        0,
        0,
        0,
        0,
    )
    rol_loss, role_accuracy, role_precision, role_recall, role_f1 = role_loss(
        role_logits,
        role_labels,
        top,
        group_roles,
        role_threshold,
        weight_role_labels,
        noise,
    )

    eps = 1e-6
    weight_rel /= (rel_f1+eps)**F1_loss_power
    # weight_sense /= sense_f1
    weight_role /= (role_f1+eps)**F1_loss_power
    # print(f"Rel Loss: {rel_loss:.4f}, Role Loss: {rol_loss:.4f}")
    total_loss = (
        weight_rel * rel_loss + weight_role * rol_loss
    )  # + weight_sense * sense_loss

    result = {
        "loss": total_loss,
        "rel_loss": rel_loss,
        "sense_loss": sense_loss,
        "role_loss": rol_loss,
        "rel_accuracy": rel_accuracy,
        "rel_precision": rel_precision,
        "rel_recall": rel_recall,
        "rel_f1": rel_f1,
        "sense_accuracy": sense_acc,
        "sense_precision": sense_precision,
        "sense_recall": sense_recall,
        "sense_f1": sense_f1,
        "role_accuracy": role_accuracy,
        "role_precision": role_precision,
        "role_recall": role_recall,
        "role_f1": role_f1,
    }

    return result


# TRAIN STEP
def train_step(
    model: SRL_MODEL,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    F1_loss_power: float,
    group_roles: bool = False,
    role_threshold: float = 0.5,
    noise: float = 0.2,
    random_sostitution_prob: float = 0,
    weight_role_labels: bool = False,
    device: str = "cuda",
):
    """
    Perform a training step

    Parameters:
        model: The model
        train_loader: The training data loader
        optimizer: The optimizer
        F1_loss_power: The power for the rescaling using the F1 score
        group_roles: Wether to group the roles to compute the loss
        role_threshold: The threshold for the role classification
        noise: The noise to add to the labels
        random_sostitution_prob: The probability of randomly substituting a token with UNK
        weight_role_labels: Wether to weight the role labels in the loss function
        device: The device to use

    Returns:
        A dictionary containing the total loss, relation loss, sense loss, role loss, relation accuracy, relation precision, relation recall, relation f1, sense accuracy, sense precision, sense recall, sense f1, role accuracy, role precision, role recall, and role f1
    """

    # Set the model to training mode
    model.train()

    total_loss = 0

    relation_loss = 0
    sense_loss = 0
    role_loss = 0

    relation_accuracy = 0
    relation_precision = 0
    relation_recall = 0
    relation_f1 = 0

    sense_accuracy = 0
    sense_precision = 0
    sense_recall = 0
    sense_f1 = 0

    role_accuracy = 0
    role_precision = 0
    role_recall = 0
    role_f1 = 0

    for batch in tqdm(train_loader, position=1):
        # text = batch['text']
        input_ids = batch["input_ids"].to(device)
        word_ids = batch["word_ids"]
        attention_masks = batch["attention_masks"].to(device)
        relations = batch["relation_position"]
        role_labels = batch["role_labels"]
        # rel_senses = batch['senses'].to(device)

        relation_label_masks = batch["relation_label_mask"].to(device)
        relation_labels = batch["relation_label"].to(device)

        senses_labels = batch["senses_labels"].to(device)

        optimizer.zero_grad()

        # random sobstitution
        if random_sostitution_prob > 0:
            rand_tensor = torch.rand(input_ids.shape).to(input_ids.device)
            mask_tensor = rand_tensor < (random_sostitution_prob * attention_masks)
            input_ids[mask_tensor] = model.tokenizer.unk_token_id

        # get the logits from the model
        relational_logits, senses_logits, role_results = model(
            input_ids, attention_masks, relations, word_ids
        )

        # compute the loss
        loss_dict = loss(
            model=model,
            rel_mask=relation_label_masks,
            rel_logits=relational_logits,
            rel_labels=relation_labels,
            sense_logits=senses_logits,
            sense_labels=senses_labels,
            role_logits=role_results,
            role_labels=role_labels,
            group_roles=group_roles,
            role_threshold=role_threshold,
            weight_role_labels=weight_role_labels,
            noise=noise,
            weight_rel=1,
            weight_sense=1,
            weight_role=1,
            F1_loss_power=F1_loss_power,
        )

        loss_dict["loss"].backward()
        clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        total_loss += loss_dict["loss"].item()

        relation_loss += loss_dict["rel_loss"].item()
        sense_loss += loss_dict["sense_loss"].item()
        role_loss += loss_dict["role_loss"].item()

        relation_accuracy += loss_dict["rel_accuracy"]
        relation_precision += loss_dict["rel_precision"]
        relation_recall += loss_dict["rel_recall"]
        relation_f1 += loss_dict["rel_f1"]

        sense_accuracy += loss_dict["sense_accuracy"]
        sense_precision += loss_dict["sense_precision"]
        sense_recall += loss_dict["sense_recall"]
        sense_f1 += loss_dict["sense_f1"]

        role_accuracy += loss_dict["role_accuracy"]
        role_precision += loss_dict["role_precision"]
        role_recall += loss_dict["role_recall"]
        role_f1 += loss_dict["role_f1"]

        # print(f"\nLoss: {loss_dict['loss'].item():.4f}, Rel Loss: {loss_dict['rel_loss'].item():.4f}, Sense Loss: {loss_dict['sense_loss'].item():.4f}, Role Loss: {loss_dict['role_loss'].item():.4f}")
        # #F1
        # print(f"Rel F1: {loss_dict['rel_f1']:.4f}, Sense F1: {loss_dict['sense_f1']:.4f}, Role F1: {loss_dict['role_f1']:.4f}")

    len_loader = len(train_loader)
    total_loss /= len_loader
    relation_loss /= len_loader
    sense_loss /= len_loader
    role_loss /= len_loader

    relation_accuracy /= len_loader
    relation_precision /= len_loader
    relation_recall /= len_loader
    relation_f1 /= len_loader

    sense_accuracy /= len_loader
    sense_precision /= len_loader
    sense_recall /= len_loader
    sense_f1 /= len_loader

    role_accuracy /= len_loader
    role_precision /= len_loader
    role_recall /= len_loader
    role_f1 /= len_loader

    return {
        "loss": total_loss,
        "rel_loss": relation_loss,
        "sense_loss": sense_loss,
        "role_loss": role_loss,
        "rel_accuracy": relation_accuracy,
        "rel_precision": relation_precision,
        "rel_recall": relation_recall,
        "rel_f1": relation_f1,
        "sense_accuracy": sense_accuracy,
        "sense_precision": sense_precision,
        "sense_recall": sense_recall,
        "sense_f1": sense_f1,
        "role_accuracy": role_accuracy,
        "role_precision": role_precision,
        "role_recall": role_recall,
        "role_f1": role_f1,
    }


def eval_step(
    model: SRL_MODEL,
    val_loader: DataLoader,
    F1_loss_power: float,
    top: bool = False,
    group_roles: bool = False,
    role_threshold: float = 0.5,
    weight_role_labels: bool = False,
    device: str = "cuda",
):
    """
    Perform an evaluation step

    Parameters:
        model: The model
        val_loader: The validation data loader
        F1_loss_power: The power for the rescaling using the F1 score
        top: Wether to use top selection for the predicted spans
        group_roles: Wether to group the roles to compute the loss
        role_threshold: The threshold for the role classification
        weight_role_labels: Wether to weight the role labels in the loss function
        device: The device to use

    Returns:
        A dictionary containing the total loss, relation loss, sense loss, role loss, relation accuracy, relation precision, relation recall, relation f1, sense accuracy, sense precision, sense recall, sense f1, role accuracy, role precision, role recall, and role f1
    """
    model.eval()

    total_loss = 0

    relation_loss = 0
    sense_loss = 0
    role_loss = 0

    relation_accuracy = 0
    relation_precision = 0
    relation_recall = 0
    relation_f1 = 0

    sense_accuracy = 0
    sense_precision = 0
    sense_recall = 0
    sense_f1 = 0

    role_accuracy = 0
    role_precision = 0
    role_recall = 0
    role_f1 = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, position=1):
            input_ids = batch["input_ids"].to(device)
            word_ids = batch["word_ids"]
            attention_masks = batch["attention_masks"].to(device)
            relations = batch["relation_position"]
            role_labels = batch["role_labels"]
            # rel_senses = batch['senses'].to(device)

            relation_label_masks = batch["relation_label_mask"].to(device)
            relation_labels = batch["relation_label"].to(device)

            senses_labels = batch["senses_labels"].to(device)

            relational_logits, senses_logits, role_results = model(
                input_ids, attention_masks, relations, word_ids
            )

            loss_dict = loss(
                model=model,
                rel_mask=relation_label_masks,
                rel_logits=relational_logits,
                rel_labels=relation_labels,
                sense_logits=senses_logits,
                sense_labels=senses_labels,
                role_logits=role_results,
                role_labels=role_labels,
                group_roles=group_roles,
                role_threshold=role_threshold,
                weight_role_labels=weight_role_labels,
                weight_rel=1,
                weight_sense=1,
                weight_role=1,
                F1_loss_power=F1_loss_power,
                noise=0,
                top=top,
            )

            total_loss += loss_dict["loss"].item()

            relation_loss += loss_dict["rel_loss"].item()
            sense_loss += loss_dict["sense_loss"].item()
            role_loss += loss_dict["role_loss"].item()

            relation_accuracy += loss_dict["rel_accuracy"]
            relation_precision += loss_dict["rel_precision"]
            relation_recall += loss_dict["rel_recall"]
            relation_f1 += loss_dict["rel_f1"]

            sense_accuracy += loss_dict["sense_accuracy"]
            sense_precision += loss_dict["sense_precision"]
            sense_recall += loss_dict["sense_recall"]
            sense_f1 += loss_dict["sense_f1"]

            role_accuracy += loss_dict["role_accuracy"]
            role_precision += loss_dict["role_precision"]
            role_recall += loss_dict["role_recall"]
            role_f1 += loss_dict["role_f1"]

    len_loader = len(val_loader)
    total_loss /= len_loader
    relation_loss /= len_loader
    sense_loss /= len_loader
    role_loss /= len_loader

    relation_accuracy /= len_loader
    relation_precision /= len_loader
    relation_recall /= len_loader
    relation_f1 /= len_loader

    sense_accuracy /= len_loader
    sense_precision /= len_loader
    sense_recall /= len_loader
    sense_f1 /= len_loader

    role_accuracy /= len_loader
    role_precision /= len_loader
    role_recall /= len_loader
    role_f1 /= len_loader

    return {
        "loss": total_loss,
        "rel_loss": relation_loss,
        "sense_loss": sense_loss,
        "role_loss": role_loss,
        "rel_accuracy": relation_accuracy,
        "rel_precision": relation_precision,
        "rel_recall": relation_recall,
        "rel_f1": relation_f1,
        "sense_accuracy": sense_accuracy,
        "sense_precision": sense_precision,
        "sense_recall": sense_recall,
        "sense_f1": sense_f1,
        "role_accuracy": role_accuracy,
        "role_precision": role_precision,
        "role_recall": role_recall,
        "role_f1": role_f1,
    }


# OTHERS
def print_and_log_results(
    result: dict, tensorboard: SummaryWriter, epoch: int, tag: str, dataset: str = "UP"
):
    """
    Print and log the results to tensorboard

    Parameters:
        result: The results
        tensorboard: The tensorboard writer
        epoch: The epoch
        tag: The tag to use (Train, Val, Test)
    """
    if dataset != "UP":
        tag = f"{dataset}/{tag}"

    print(f"\n\t{tag} Loss: {result['loss']:.4f}")
    print(
        f"\t{tag} Rel Loss: {result['rel_loss']:.4f}, accuracy: {result['rel_accuracy']:.4f}, precision: {result['rel_precision']:.4f}, recall: {result['rel_recall']:.4f}, f1: {result['rel_f1']:.4f}"
    )
    # print(f"\t{tag} Sense Loss: {result['sense_loss']:.4f}, accuracy: {result['sense_accuracy']:.4f}, precision: {result['sense_precision']:.4f}, recall: {result['sense_recall']:.4f}, f1: {result['sense_f1']:.4f}")
    print(
        f"\t{tag} Role Loss: {result['role_loss']:.4f}, accuracy: {result['role_accuracy']:.4f}, precision: {result['role_precision']:.4f}, recall: {result['role_recall']:.4f}, f1: {result['role_f1']:.4f}"
    )

    tensorboard.add_scalar(f"Loss/{tag}", result["loss"], epoch)

    tensorboard.add_scalar(f"Rel Loss/{tag}", result["rel_loss"], epoch)
    tensorboard.add_scalar(f"Rel Acc/{tag}", result["rel_accuracy"], epoch)
    tensorboard.add_scalar(f"Rel Precision/{tag}", result["rel_precision"], epoch)
    tensorboard.add_scalar(f"Rel Recall/{tag}", result["rel_recall"], epoch)
    tensorboard.add_scalar(f"Rel F1/{tag}", result["rel_f1"], epoch)

    # tensorboard.add_scalar(f'Sense Loss/{tag}', result['sense_loss'], epoch)
    # tensorboard.add_scalar(f'Sense Acc/{tag}', result['sense_accuracy'], epoch)
    # tensorboard.add_scalar(f'Sense Precision/{tag}', result['sense_precision'], epoch)
    # tensorboard.add_scalar(f'Sense Recall/{tag}', result['sense_recall'], epoch)
    # tensorboard.add_scalar(f'Sense F1/{tag}', result['sense_f1'], epoch)

    tensorboard.add_scalar(f"Role Loss/{tag}", result["role_loss"], epoch)
    tensorboard.add_scalar(f"Role Acc/{tag}", result["role_accuracy"], epoch)
    tensorboard.add_scalar(f"Role Precision/{tag}", result["role_precision"], epoch)
    tensorboard.add_scalar(f"Role Recall/{tag}", result["role_recall"], epoch)
    tensorboard.add_scalar(f"Role F1/{tag}", result["role_f1"], epoch)


# MAIN TRAIN FUNCTION
def train(
    model: SRL_MODEL,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 100,
    init_lr: float = 1e-3,
    lr_encoder: float = 1e-5,
    F1_loss_power: float = 2,
    patience: int = 5,
    noise: float = 0,
    random_sostitution_prob: float = 0,
    role_threshold: float = 0.5,
    group_roles: bool = False,
    top: bool = False,
    weight_role_labels: bool = False,
    device: torch.device = "cuda",
    name: str = "SRL",
    dataset: str = "UP",
):
    """
    Train the model

    Parameters:
        model: The model
        train_loader: The training data loader
        val_loader: The validation data loader
        test_loader: The test data loader
        epochs: The number of epochs
        init_lr: The initial learning rate
        lr_encoder: The learning rate for the encoder
        F1_loss_power: The power for the rescaling using the F1 score
        patience: The patience for the early stopping
        noise: Final noise to add to the labels, will be incremented linearly from 0
        random_sostitution_prob: The probability of randomly substituting a token with UNK
        role_threshold: The threshold for the role classification
        group_roles: Wether to group the roles to compute the loss
        top: Wether to use top selection for the predicted spans in the role classification for evaluation
        weight_role_labels: Wether to weight the role labels in the loss function
        device: The device to use
        name: The name of the model, used for logging on tensorboard
        dataset: The dataset to use (UP or NOM)
    """

    tensorboard = SummaryWriter(log_dir=f"runs/{name}")

    optimizer = optim.AdamW(
        [
            {"params": model.bert.parameters(), "lr": lr_encoder},
            {
                "params": [
                    p[1] for p in model.named_parameters() if "bert" not in p[0]
                ],
                "lr": init_lr,
            },
        ],
        lr=init_lr,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=0.1
    )

    patience_counter = 0
    best_val_loss = float("inf")
    for epoch in tqdm(range(34,epochs)):
        train_result = train_step(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            F1_loss_power=F1_loss_power,
            weight_role_labels=weight_role_labels,
            group_roles=group_roles,
            role_threshold=role_threshold,
            noise=noise,
            random_sostitution_prob=random_sostitution_prob,
            device=device,
        )
        val_result = eval_step(
            model=model,
            val_loader=val_loader,
            F1_loss_power=F1_loss_power,
            weight_role_labels=weight_role_labels,
            top=top,
            group_roles=group_roles,
            role_threshold=role_threshold,
            device=device,
        )

        print()
        print_and_log_results(train_result, tensorboard, epoch, "Train", dataset)
        print()
        print_and_log_results(val_result, tensorboard, epoch, "Val", dataset)

        tensorboard.add_scalar(
            "Learning Rate/encoder", optimizer.param_groups[0]["lr"], epoch
        )
        tensorboard.add_scalar(
            "Learning Rate/other", optimizer.param_groups[1]["lr"], epoch
        )

        scheduler.step(val_result["loss"])

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), f"models/{name}.pt")
        else:
            if patience_counter == patience:
                break
            patience_counter += 1

    model.load_state_dict(torch.load(f"models/{name}.pt"))

    final_result = eval_step(
        model,
        train_loader,
        F1_loss_power=F1_loss_power,
        group_roles=group_roles,
        role_threshold=role_threshold,
        device=device,
    )
    final_val_result = eval_step(
        model,
        val_loader,
        F1_loss_power=F1_loss_power,
        group_roles=group_roles,
        role_threshold=role_threshold,
        device=device,
    )
    test_result = eval_step(
        model,
        test_loader,
        F1_loss_power=F1_loss_power,
        group_roles=group_roles,
        role_threshold=role_threshold,
        device=device,
    )
    print_and_log_results(final_result, tensorboard, epochs, "Train", dataset)
    print_and_log_results(final_val_result, tensorboard, epochs, "Val", dataset)
    print_and_log_results(test_result, tensorboard, epochs, "Test", dataset)

    tensorboard.close()
