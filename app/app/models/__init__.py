from app.models.baseline import BaselineNet
from app.models.personalized_user_id import PerUserIDNet
from app.models.personalized_hubi_medium import PerHuBiMedNet

models = {
    'baseline_sgl': BaselineNet,
    'baseline_avg': BaselineNet,
    'personalized_user_id': PerUserIDNet,
    'personalized_hubi_med': PerHuBiMedNet
}
