document.getElementById('customFile').addEventListener('change', function(event){
    var reader = new FileReader();
    reader.onload = function(){
        var output = document.getElementById('imagePreview');
        output.innerHTML = `<img src="${reader.result}" alt="Image preview"/>`;
    };
    reader.readAsDataURL(event.target.files[0]);
});

document.getElementById('upload-form').addEventListener('submit', function(e){
    e.preventDefault();
    var form = document.getElementById('upload-form');
    var formData = new FormData(form);

    document.getElementById('message').textContent = 'Predicting...';

    fetch('/predict', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById('message').textContent = data;
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

document.querySelector('.custom-file-input').addEventListener('change', function (e) {
    var fileName = e.target.files[0].name; // Get the file name
    var nextSibling = e.target.nextElementSibling; // Get the label element
    nextSibling.innerText = fileName; // Update the label text
});
