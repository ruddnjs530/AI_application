from flask import Flask, request, render_template, url_for, jsonify
from werkzeug.utils import secure_filename
import os
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import subprocess
from model import Generator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator_AB = Generator(input_nc=3, output_nc=3).to(device)
generator_AB.load_state_dict(torch.load("cycle-gan-model/generator_AB_final.pth", map_location=device))
generator_AB.eval()

books = [
    {"id": 1, "title": "Book 1"},
    {"id": 2, "title": "Book 2"},
    {"id": 3, "title": "Book 3"}
]

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/chatgpt", methods=["GET"])
def chatgpt():
    return render_template("chat_gpt.html")

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return "No image selected!"

    image_file = request.files['image']
    if image_file.filename == '':
        return "No image selected!"

    if image_file.filename.lower().endswith(('.png', '.jpg', '.gif', '.jpeg')):
        filename = secure_filename(image_file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        image_data = io.BytesIO(image_file.read())
        image = Image.open(image_data).convert("RGB")

        image.save(save_path, format='JPEG', quality=95)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            fake_image_tensor = generator_AB(image_tensor)
            fake_image_tensor = fake_image_tensor.squeeze().cpu().detach()

            fake_image_tensor = fake_image_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            fake_image_tensor = torch.clamp(fake_image_tensor, 0, 1)
            fake_image = transforms.ToPILImage()(fake_image_tensor)

        fake_image = fake_image.resize(image.size)

        fake_save_path = os.path.join(app.config['UPLOAD_FOLDER'], "fake_" + filename)
        fake_image.save(fake_save_path, format='JPEG', quality=95)

        original_image_url = url_for('static', filename='images/' + filename)
        fake_image_url = url_for('static', filename='images/' + "fake_" + filename)

        return render_template("display_image.html", original_image_url=original_image_url, fake_image_url=fake_image_url)
    else:
        return "Invalid image format!"

@app.route("/app", methods=["GET"])
def myApp():
    message = request.args.get("message")
    return render_template("chat_gpt.html", msg=message)

@app.route("/process", methods=["GET"])
def proc():
    try:
        process = subprocess.run(['pip', 'install', 'subprocess'], capture_output=True, text=True)
        output = process.stdout
    except subprocess.CalledProcessError as e:
        output = f"Error: {str(e)}"
    return render_template("chat_gpt.html", msg=output)

@app.route("/predict", methods=["GET"])
def predict():
    data = request.args.get("data")

    input_tensor = torch.tensor([float(data)], device=device)

    with torch.no_grad():
        prediction_tensor = input_tensor * 2
        prediction = prediction_tensor.item()

    return jsonify({'user_id': 'bseo', 'prediction': {
        'result': prediction
    }})

@app.route("/books", methods=["GET"])
def get_books():
    return jsonify(books)

@app.route("/books/<int:book_id>", methods=["GET"])
def get_book(book_id):
    book = [b for b in books if b["id"] == book_id]
    if book:
        return jsonify(book[0])
    else:
        return jsonify({"message": "no book found"}), 404

if __name__ == "__main__":
    app.run(debug=True)