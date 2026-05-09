package com.ajay.wasteclassifier

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * WasteClassifier — Phase 3 Android Activity
 * Author: Pulyala Ajay Kumar
 *
 * Features:
 *  - Real-time camera classification via CameraX ImageAnalysis
 *  - Gallery upload fallback
 *  - Offline TFLite inference (no internet required)
 *  - 10-class waste categorisation with disposal tips
 */
class MainActivity : AppCompatActivity() {

    // ── UI refs ──────────────────────────────────────────────────────────────
    private lateinit var previewView: androidx.camera.view.PreviewView
    private lateinit var resultLabel: TextView
    private lateinit var confidenceBar: ProgressBar
    private lateinit var confidenceText: TextView
    private lateinit var disposalTip: TextView
    private lateinit var captureBtn: ImageButton
    private lateinit var galleryBtn: Button
    private lateinit var overlayView: ClassificationOverlayView

    // ── ML ───────────────────────────────────────────────────────────────────
    private lateinit var tfliteInterpreter: Interpreter
    private lateinit var labels: List<String>

    // ── Camera ───────────────────────────────────────────────────────────────
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null

    companion object {
        private const val IMG_SIZE = 224
        private const val GALLERY_REQUEST = 1001
        private const val CONFIDENCE_THRESHOLD = 0.55f

        private val DISPOSAL_TIPS = mapOf(
            "cardboard"  to "Flatten and place in dry paper recycling bin.",
            "e_waste"    to "Take to a certified e-waste collection centre. Never bin.",
            "food_waste" to "Add to compost bin or wet waste collection.",
            "glass"      to "Rinse and place in glass recycling. Remove lids.",
            "hazardous"  to "Take to a hazardous waste facility. Do NOT mix with general waste.",
            "metal"      to "Rinse cans. Place in dry recycling or take to scrap dealer.",
            "paper"      to "Keep dry. Add to paper recycling or newspaper pickup.",
            "plastic"    to "Rinse. Check resin code and place in plastic recycling.",
            "rubber"     to "Contact tyre/rubber recycler. Do not burn.",
            "textile"    to "Donate wearable clothes. Take unusable fabric to textile recycler."
        )
    }

    // ── Lifecycle ────────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        loadModel()
        loadLabels()
        cameraExecutor = Executors.newSingleThreadExecutor()
        startCamera()

        captureBtn.setOnClickListener { takePhoto() }
        galleryBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, GALLERY_REQUEST)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        tfliteInterpreter.close()
    }

    // ── Model loading ────────────────────────────────────────────────────────

    private fun loadModel() {
        val assetFd = assets.openFd("waste_classifier_dynamic.tflite")
        val inputStream = FileInputStream(assetFd.fileDescriptor)
        val fileChannel = inputStream.channel
        val model: MappedByteBuffer = fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            assetFd.startOffset,
            assetFd.declaredLength
        )
        val options = Interpreter.Options().apply { numThreads = 4 }
        tfliteInterpreter = Interpreter(model, options)
    }

    private fun loadLabels() {
        labels = assets.open("labels.txt").bufferedReader().readLines()
    }

    // ── Camera ───────────────────────────────────────────────────────────────

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
                .also { it.setSurfaceProvider(previewView.surfaceProvider) }

            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(IMG_SIZE, IMG_SIZE))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also { analysis ->
                    analysis.setAnalyzer(cameraExecutor) { imageProxy ->
                        val bitmap = imageProxy.toBitmap()
                        if (bitmap != null) {
                            runOnUiThread { classify(bitmap) }
                        }
                        imageProxy.close()
                    }
                }

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview, imageCapture, imageAnalyzer
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        // Freeze current frame for a stable result
        imageCapture?.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = image.toBitmap()
                    if (bitmap != null) classify(bitmap)
                    image.close()
                }
                override fun onError(exc: ImageCaptureException) {
                    Toast.makeText(
                        this@MainActivity,
                        "Capture failed: ${exc.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        )
    }

    // ── Inference ────────────────────────────────────────────────────────────

    private fun classify(bitmap: Bitmap) {
        val resized = Bitmap.createScaledBitmap(bitmap, IMG_SIZE, IMG_SIZE, true)

        // Build input tensor [1, 224, 224, 3] float32
        val inputBuffer = FloatArray(1 * IMG_SIZE * IMG_SIZE * 3)
        var idx = 0
        for (y in 0 until IMG_SIZE) {
            for (x in 0 until IMG_SIZE) {
                val pixel = resized.getPixel(x, y)
                inputBuffer[idx++] = ((pixel shr 16) and 0xFF) / 255f   // R
                inputBuffer[idx++] = ((pixel shr  8) and 0xFF) / 255f   // G
                inputBuffer[idx++] = ( pixel         and 0xFF) / 255f   // B
            }
        }

        val inputArray = Array(1) {
            Array(IMG_SIZE) { y ->
                Array(IMG_SIZE) { x ->
                    FloatArray(3).also { ch ->
                        val pixel = resized.getPixel(x, y)
                        ch[0] = ((pixel shr 16) and 0xFF) / 255f
                        ch[1] = ((pixel shr  8) and 0xFF) / 255f
                        ch[2] = ( pixel         and 0xFF) / 255f
                    }
                }
            }
        }

        val outputBuffer = Array(1) { FloatArray(labels.size) }
        tfliteInterpreter.run(inputArray, outputBuffer)
        val probs = outputBuffer[0]

        val topIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
        val topClass = labels[topIdx]
        val topConf  = probs[topIdx]

        updateUI(topClass, topConf, probs)
        logLowConfidence(topClass, topConf)
    }

    // ── UI update ────────────────────────────────────────────────────────────

    private fun updateUI(topClass: String, confidence: Float, probs: FloatArray) {
        val label = topClass.replace("_", " ")
            .split(" ").joinToString(" ") { it.replaceFirstChar(Char::titlecase) }

        resultLabel.text = label
        confidenceBar.progress = (confidence * 100).toInt()
        confidenceText.text = "%.1f%%".format(confidence * 100)

        if (confidence < CONFIDENCE_THRESHOLD) {
            resultLabel.setTextColor(getColor(R.color.warning_amber))
            disposalTip.text = "⚠️ Low confidence — try better lighting or a closer shot."
        } else {
            resultLabel.setTextColor(getColor(R.color.text_primary))
            disposalTip.text = DISPOSAL_TIPS[topClass] ?: "Consult your local waste authority."
        }

        overlayView.updateResults(labels.zip(probs.toList()))
    }

    // ── Low-confidence logging (for active learning Phase 4) ─────────────────

    private fun logLowConfidence(className: String, confidence: Float) {
        if (confidence < CONFIDENCE_THRESHOLD) {
            val logFile = getExternalFilesDir(null)?.resolve("low_confidence_log.csv")
            logFile?.appendText("${System.currentTimeMillis()},$className,${"%.4f".format(confidence)}\n")
        }
    }

    // ── Gallery result ───────────────────────────────────────────────────────

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == GALLERY_REQUEST && resultCode == RESULT_OK) {
            val uri: Uri = data?.data ?: return
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            if (bitmap != null) classify(bitmap)
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private fun bindViews() {
        previewView    = findViewById(R.id.previewView)
        resultLabel    = findViewById(R.id.resultLabel)
        confidenceBar  = findViewById(R.id.confidenceBar)
        confidenceText = findViewById(R.id.confidenceText)
        disposalTip    = findViewById(R.id.disposalTip)
        captureBtn     = findViewById(R.id.captureBtn)
        galleryBtn     = findViewById(R.id.galleryBtn)
        overlayView    = findViewById(R.id.overlayView)
    }

    // Convert ImageProxy to Bitmap (CameraX utility)
    private fun ImageProxy.toBitmap(): Bitmap? {
        val buffer = planes[0].buffer
        val bytes = ByteArray(buffer.remaining())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
}
