import requests
from datetime import datetime

BOT_TOKEN = '7754333051:AAEnVQL5XK-c15gi2Ap8GySrLJURLB4XTNc'
CHAT_ID = '7743560797'

def send_violation_alert(image_path, car_id):
    caption = f"🚨 Vi phạm: Xe ID {car_id} lúc {datetime.now().strftime('%H:%M:%S')}"
    with open(image_path, 'rb') as photo:
        requests.post(
            f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto',
            data={'chat_id': CHAT_ID, 'caption': caption},
            files={'photo': photo}
        )
