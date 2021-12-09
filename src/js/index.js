function openCVReady() {
  cv['onRuntimeInitialized']=()=>{
    // do all your work here

    let video = document.getElementById("videoInput"); // video is the id of video tag
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log("An error occurred! " + err);
    });

    //Remove Default Video Block
    video.style.display = "none"

    //Initialize Necessary Variables
    const FPS = 30
    let height = video.height
    let width = video.width

    let src = new cv.Mat(height, width, cv.CV_8UC4);
    let dst = new cv.Mat(height, width, cv.CV_8UC1);
    let cap = new cv.VideoCapture(video);

    let gray = new cv.Mat();
    let faces = new cv.RectVector();
    let utils = new Utils("errorMessage")

    let classifier = new cv.CascadeClassifier();
    let faceCascadeFile = 'src/js/haarcascade_frontalface_default.xml';

    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        classifier.load(faceCascadeFile);
        console.log("Cascade XML Loaded");
    });
    var emotionModel = '';
    
    tf.loadGraphModel('models/js/efficientB0_Affect_Graph/model.json').then((loadedModel) => {
      emotionModel = loadedModel;
  });

    var mapper = {
        0: 'Neutral',
        1: 'Happy',
        2: 'Sad',
        3: 'Surprise',
        4: 'Fear',
        5: 'Disgust',
        6: 'Anger',
        7: 'Contempt'
             };

    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        // cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        let msize = new cv.Size(0, 0);
        let dsize = new cv.Size(224, 224);
        try {
            classifier.detectMultiScale(gray, faces, 1.15, 3, 0, msize, msize); 
            for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                let roiSrc = src.roi(face);
                let sample = new cv.Mat();
                let resizedImage = new cv.Mat();
                
                cv.cvtColor(roiSrc, sample, cv.COLOR_RGBA2RGB, 0);
                cv.resize(sample, resizedImage, dsize, 0, 0, cv.INTER_AREA);

                // let inputTensor = tf.reshape(tf.tensor(resizedImage.data), [1,224,224,3]);

                // let outputTensor = emotionModel.predict(inputTensor);
                const outputTensor = tf.tidy(() => {
                    // Get pixels data from an image.
                    let inputTensor = tf.cast(tf.reshape(tf.tensor(resizedImage.data), [1,224,224,3]), 'float32');
                    // Run the inference.
                    let outputTensor = emotionModel.predict(inputTensor);
                    return outputTensor
                });

                let pred = outputTensor[1].argMax(axis=1).arraySync()[0]
                let emotionLabel = mapper[pred]
                let confidence = Math.round((100*outputTensor[1].arraySync()[0][pred] + Number.EPSILON) * 100) / 100
                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                cv.rectangle(dst, point1, point2, [255, 0, 0, 255], 2);
                cv.putText(dst, emotionLabel + ': ' + confidence + '%', new cv.Point(face.x, face.y-10),
                 cv.FONT_HERSHEY_SIMPLEX, 0.9, [255,0,255, 0], 2);
            }
        } catch(error) {
        // console.error(error)
    }

    cv.imshow("canvas", dst);
    // schedule next one.
    let delay = 1000/FPS - (Date.now() - begin);
    setTimeout(processVideo, delay);
}
// schedule first one.
setTimeout(processVideo, 0);

};
}


