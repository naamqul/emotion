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
    let sample = new cv.Mat();

    let classifier = new cv.CascadeClassifier();
    let faceCascadeFile = 'haarcascade_frontalface_default.xml';

    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        classifier.load(faceCascadeFile);
        console.log("Cascade XML Loaded");
    });
    const emotionModel = tflite.loadTFLiteModel('models/efficientB0_Affect.tflite');
    function processVideo() {
        let begin = Date.now();
        cap.read(src);
        src.copyTo(dst);
        cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
        // cv.cvtColor(src, dst, cv.COLOR_RGBA2GRAY);
        let msize = new cv.Size(0, 0);

        try {
            classifier.detectMultiScale(gray, faces, 1.15, 3, 0, msize, msize); 
            for (let i = 0; i < faces.size(); ++i) {
                let face = faces.get(i);
                let roiSrc = src.roi(face);
                cv.cvtColor(roiSrc, sample, cv.COLOR_RGBA2RGB, 0);
                sample = tf.fromPixels(sample).resizeBilinear([224, 224, 3])
                sample.reshape([1,224,224,3])
                let outputTensor = emotionModel.predict(sample) as tf.Tensor;
                console.log(outputTensor)
                let point1 = new cv.Point(face.x, face.y);
                let point2 = new cv.Point(face.x + face.width, face.y + face.height);
                cv.rectangle(dst, point1, point2, [255, 0, 0, 255]);
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


