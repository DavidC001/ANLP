import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


def relation_loss(mask: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor):
    # Compute loss for relational classification
    logits = logits * mask

    # Calculate positive weight for relational classification
    pos_weight_rel = (labels == 0).float().sum() / (labels == 1).float().sum()

    # Use BCEWithLogitsLoss with pos_weight
    loss_function_relation_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight_rel.to(logits.device), reduction='mean')
    relational_loss = loss_function_relation_weighted(logits, labels.to(logits.device).float())

    # Compute accuracy and F1 score for relational classification
    with torch.no_grad():
        rel_preds = (torch.sigmoid(logits) > 0.5).float().flatten()
        rel_labels = labels.float().flatten()
        rel_accuracy = (rel_preds == rel_labels).float().mean()

        rel_labels = rel_labels.cpu().numpy()
        rel_preds = rel_preds.cpu().numpy()
        rel_precision, rel_recall, rel_f1, _ = precision_recall_fscore_support(rel_labels, rel_preds, average='binary', zero_division=1)

    return relational_loss, rel_accuracy, rel_precision, rel_recall, rel_f1

def senses_loss(logits: torch.Tensor, labels: torch.Tensor):
    # Compute loss for sense classification
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(logits, labels)
    # loss /= logits.size(0)

    # Compute accuracy and F1 score for sense classification
    with torch.no_grad():
        sense_preds = torch.argmax(logits, dim=-1)
        labels = torch.argmax(labels, dim=-1)
        sense_acc = (sense_preds == labels).float().mean()
        sense_preds = sense_preds.cpu().numpy()
        labels = labels.cpu().numpy()
        # breakpoint()
        sense_precision, sense_recall, sense_f1, _ = precision_recall_fscore_support(labels, sense_preds, average='weighted', zero_division=1)

    return loss, sense_acc, sense_precision, sense_recall, sense_f1

def role_loss(results: list[torch.Tensor], labels: list[torch.Tensor]):
    # Compute loss for role classification
    role_loss = 0
    all_role_preds = []
    all_role_labels = []
    correct_roles = 0
    total_roles = 0

    for i in range(len(results)):
        role_logits = results[i]
        role_labels = labels[i].to(role_logits.device).float()

        # breakpoint()
        # Calculate positive weight for role classification for each role
        pos_weight = torch.tensor([max(1,(role_labels[:,:,i] == 0).float().sum().item()) / max(1,(role_labels[:,:,i] == 1).float().sum().item()) for i in range(role_labels.shape[2])]).to(role_logits.device)
        # Have all the roles to have the same weight for the loss

        # Weighted Binary Cross-Entropy Loss
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        loss = criterion(role_logits, role_labels).view(-1, role_labels.shape[-1]).mean(0).sum()
        role_loss += loss

        # Compute accuracy and F1 score for role classification
        role_preds = (torch.sigmoid(role_logits) > 0.5).float()
        
        # breakpoint()
        for j in range(role_labels.shape[0]):
            for k in range(role_labels.shape[1]):
                all_role_labels.append(role_labels[j][k].cpu().numpy())
                all_role_preds.append(role_preds[j][k].cpu().numpy())
        
    role_loss /= len(results)
    
    # compute accuracy for each role
    for i in range(len(all_role_labels)):
        for j in range(len(all_role_labels[i])):
            if all_role_preds[i][j] == all_role_labels[i][j]:
                correct_roles += 1
            total_roles += 1
    role_accuracy = correct_roles / total_roles

    # compute precision, recall and f1 for each role
    # breakpoint()
    role_precision, role_recall, role_f1, _ = precision_recall_fscore_support(all_role_labels, all_role_preds, zero_division=1)
    role_precision = role_precision.mean()
    role_recall = role_recall.mean()
    role_f1 = role_f1.mean()

    return role_loss, role_accuracy, role_precision, role_recall, role_f1

def loss(rel_mask: torch.Tensor, rel_logits: torch.Tensor, rel_labels: torch.Tensor, 
         sense_logits: torch.Tensor, sense_labels: torch.Tensor, 
         role_logits: torch.Tensor, role_labels: torch.Tensor,
         model: nn.Module, l2_lambda: float=0.001,
         weight_rel: float=1, weight_sense: float=1, weight_role: float=1,
         ):
    
    # Compute loss for each task
    rel_loss, rel_accuracy, rel_precision, rel_recall, rel_f1 = relation_loss(rel_mask, rel_logits, rel_labels)
    # sense_loss, sense_acc, sense_precision, sense_recall, sense_f1 = senses_loss(sense_logits, sense_labels)
    sense_loss, sense_acc, sense_precision, sense_recall, sense_f1 = torch.tensor(0), 0, 0, 0, 0
    rol_loss, role_accuracy, role_precision, role_recall, role_f1 = role_loss(role_logits, role_labels)

    weight_rel /= rel_f1
    # weight_sense /= sense_f1
    weight_role /= role_f1
    # print(f"Rel Loss: {rel_loss:.4f}, Role Loss: {rol_loss:.4f}")
    total_loss = weight_rel * rel_loss + weight_role * rol_loss # + weight_sense * sense_loss
    # should help avoid having classes always be predicted, do not count the parameters of model.bert
    total_loss += l2_lambda * sum(p[1].pow(2.0).sum() for p in model.named_parameters() if 'bert' not in p[0])

    result = {
        "loss": total_loss,
        "rel_loss": rel_loss, 
        "sense_loss": sense_loss, 
        "role_loss": rol_loss,

        "rel_accuracy": rel_accuracy, "rel_precision": rel_precision, "rel_recall": rel_recall, "rel_f1": rel_f1,
        
        "sense_accuracy": sense_acc, "sense_precision": sense_precision, "sense_recall": sense_recall, "sense_f1": sense_f1,
        
        "role_accuracy": role_accuracy, "role_precision": role_precision, "role_recall": role_recall, "role_f1": role_f1
    }

    return result



def train_step(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, l2_lambda: float, device: torch.device):

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
        input_ids = batch['input_ids'].to(device)
        word_ids = batch['word_ids']
        attention_masks = batch['attention_masks'].to(device)
        relations = batch['relation_position']
        role_labels = batch['role_labels']
        # rel_senses = batch['senses'].to(device)

        relation_label_masks = batch['relation_label_mask'].to(device)
        relation_labels = batch['relation_label'].to(device)

        senses_labels = batch['senses_labels'].to(device)

        optimizer.zero_grad()

        # breakpoint()

        relational_logits, senses_logits, role_results = model(input_ids, attention_masks, relations, word_ids)

        loss_dict = loss(relation_label_masks, relational_logits, relation_labels, 
                         senses_logits, senses_labels, 
                         role_results, role_labels,
                         model, l2_lambda=l2_lambda,
                         weight_rel=1, weight_sense=1, weight_role=1)

        loss_dict['loss'].backward()
        optimizer.step()

        total_loss += loss_dict['loss'].item()

        relation_loss += loss_dict['rel_loss'].item()
        sense_loss += loss_dict['sense_loss'].item()
        role_loss += loss_dict['role_loss'].item()

        relation_accuracy += loss_dict['rel_accuracy']
        relation_precision += loss_dict['rel_precision']
        relation_recall += loss_dict['rel_recall']
        relation_f1 += loss_dict['rel_f1']

        sense_accuracy += loss_dict['sense_accuracy']
        sense_precision += loss_dict['sense_precision']
        sense_recall += loss_dict['sense_recall']
        sense_f1 += loss_dict['sense_f1']

        role_accuracy += loss_dict['role_accuracy']
        role_precision += loss_dict['role_precision']
        role_recall += loss_dict['role_recall']
        role_f1 += loss_dict['role_f1']

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
        "rel_loss": relation_loss, "sense_loss": sense_loss, "role_loss": role_loss,

        "rel_accuracy": relation_accuracy, "rel_precision": relation_precision, "rel_recall": relation_recall, "rel_f1": relation_f1,
        
        "sense_accuracy": sense_accuracy, "sense_precision": sense_precision, "sense_recall": sense_recall, "sense_f1": sense_f1,
        
        "role_accuracy": role_accuracy, "role_precision": role_precision, "role_recall": role_recall, "role_f1": role_f1
    }

def eval_step(model: nn.Module, val_loader: DataLoader, l2_lambda: float, device: torch.device):
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
            input_ids = batch['input_ids'].to(device)
            word_ids = batch['word_ids']
            attention_masks = batch['attention_masks'].to(device)
            relations = batch['relation_position']
            role_labels = batch['role_labels']
            # rel_senses = batch['senses'].to(device)

            relation_label_masks = batch['relation_label_mask'].to(device)
            relation_labels = batch['relation_label'].to(device)

            senses_labels = batch['senses_labels'].to(device)

            relational_logits, senses_logits, role_results = model(input_ids, attention_masks, relations, word_ids)

            loss_dict = loss(relation_label_masks, relational_logits, relation_labels, 
                             senses_logits, senses_labels, 
                             role_results, role_labels,
                             model, l2_lambda=l2_lambda,
                             weight_rel=1, weight_sense=1, weight_role=1)

            total_loss += loss_dict['loss'].item()

            relation_loss += loss_dict['rel_loss'].item()
            sense_loss += loss_dict['sense_loss'].item()
            role_loss += loss_dict['role_loss'].item()

            relation_accuracy += loss_dict['rel_accuracy']
            relation_precision += loss_dict['rel_precision']
            relation_recall += loss_dict['rel_recall']
            relation_f1 += loss_dict['rel_f1']

            sense_accuracy += loss_dict['sense_accuracy']
            sense_precision += loss_dict['sense_precision']
            sense_recall += loss_dict['sense_recall']
            sense_f1 += loss_dict['sense_f1']

            role_accuracy += loss_dict['role_accuracy']
            role_precision += loss_dict['role_precision']
            role_recall += loss_dict['role_recall']
            role_f1 += loss_dict['role_f1']

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
        "rel_loss": relation_loss, "sense_loss": sense_loss, "role_loss": role_loss,

        "rel_accuracy": relation_accuracy, "rel_precision": relation_precision, "rel_recall": relation_recall, "rel_f1": relation_f1,
        
        "sense_accuracy": sense_accuracy, "sense_precision": sense_precision, "sense_recall": sense_recall, "sense_f1": sense_f1,
        
        "role_accuracy": role_accuracy, "role_precision": role_precision, "role_recall": role_recall, "role_f1": role_f1
    }


def print_and_log_results(result: dict, tensorboard: SummaryWriter, epoch: int, tag: str):
    print(f"\n\t{tag} Loss: {result['loss']:.4f}")
    print(f"\t{tag} Rel Loss: {result['rel_loss']:.4f}, accuracy: {result['rel_accuracy']:.4f}, precision: {result['rel_precision']:.4f}, recall: {result['rel_recall']:.4f}, f1: {result['rel_f1']:.4f}")
    # print(f"\t{tag} Sense Loss: {result['sense_loss']:.4f}, accuracy: {result['sense_accuracy']:.4f}, precision: {result['sense_precision']:.4f}, recall: {result['sense_recall']:.4f}, f1: {result['sense_f1']:.4f}")
    print(f"\t{tag} Role Loss: {result['role_loss']:.4f}, accuracy: {result['role_accuracy']:.4f}, precision: {result['role_precision']:.4f}, recall: {result['role_recall']:.4f}, f1: {result['role_f1']:.4f}")

    tensorboard.add_scalar(f'Loss/{tag}', result['loss'], epoch)
    
    tensorboard.add_scalar(f'Rel Loss/{tag}', result['rel_loss'], epoch)
    tensorboard.add_scalar(f'Rel Acc/{tag}', result['rel_accuracy'], epoch)
    tensorboard.add_scalar(f'Rel Precision/{tag}', result['rel_precision'], epoch)
    tensorboard.add_scalar(f'Rel Recall/{tag}', result['rel_recall'], epoch)
    tensorboard.add_scalar(f'Rel F1/{tag}', result['rel_f1'], epoch)
    
    # tensorboard.add_scalar(f'Sense Loss/{tag}', result['sense_loss'], epoch)
    # tensorboard.add_scalar(f'Sense Acc/{tag}', result['sense_accuracy'], epoch)
    # tensorboard.add_scalar(f'Sense Precision/{tag}', result['sense_precision'], epoch)
    # tensorboard.add_scalar(f'Sense Recall/{tag}', result['sense_recall'], epoch)
    # tensorboard.add_scalar(f'Sense F1/{tag}', result['sense_f1'], epoch)

    tensorboard.add_scalar(f'Role Loss/{tag}', result['role_loss'], epoch)
    tensorboard.add_scalar(f'Role Acc/{tag}', result['role_accuracy'], epoch)
    tensorboard.add_scalar(f'Role Precision/{tag}', result['role_precision'], epoch)
    tensorboard.add_scalar(f'Role Recall/{tag}', result['role_recall'], epoch)
    tensorboard.add_scalar(f'Role F1/{tag}', result['role_f1'], epoch)
    

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
        epochs: int=100, init_lr: float=1e-3, lr_encoder: float=1e-5, l2_lambda: float=1e-5,
        device: torch.device="cuda", name: str="SRL"):
    tensorboard = SummaryWriter(log_dir=f'runs/{name}')

    optimizer = optim.AdamW([
            {'params': model.bert.parameters(), 'lr': lr_encoder},
            {'params': [p[1] for p in model.named_parameters() if 'bert' not in p[0]], 'lr': init_lr}
        ], lr=init_lr, weight_decay=0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00001)

    for epoch in tqdm(range(epochs)):
        train_result = train_step(model, train_loader, optimizer, l2_lambda, device)
        val_result = eval_step(model, val_loader, l2_lambda, device)

        print()
        print_and_log_results(train_result, tensorboard, epoch, "Train")
        print()
        print_and_log_results(val_result, tensorboard, epoch, "Val")

        tensorboard.add_scalar('Learning Rate/encoder', optimizer.param_groups[0]['lr'], epoch)
        tensorboard.add_scalar('Learning Rate/other', optimizer.param_groups[1]['lr'], epoch)

        scheduler.step()

    final_result = eval_step(model, train_loader, l2_lambda=l2_lambda, device=device)
    final_val_result = eval_step(model, val_loader, l2_lambda=l2_lambda, device=device)
    test_result = eval_step(model, test_loader, l2_lambda=l2_lambda, device=device)
    print_and_log_results(final_result, tensorboard, epochs, "Train")
    print_and_log_results(final_val_result, tensorboard, epochs, "Val")
    print_and_log_results(test_result, tensorboard, epochs, "Test")

    tensorboard.close()
