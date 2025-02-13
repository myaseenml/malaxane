$(document).ready(function () {
    $('#uploadForm').on('submit', function (e) {
        e.preventDefault();
        var formData = new FormData();
        formData.append('carImage', $('#carImage')[0].files[0]);
        formData.append('bgImage', $('#bgImage')[0].files[0]);

        $.ajax({
            url: '/process',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function (response) {
                $('#outputImage').attr('src', '/get_image');
                $('#result').removeClass('d-none');
            },
            error: function () {
                alert('Error processing images. Please try again.');
            }
        });
    });
});