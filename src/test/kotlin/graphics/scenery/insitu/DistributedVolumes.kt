package graphics.scenery.insitu

import graphics.scenery.*
import graphics.scenery.backends.Renderer
import graphics.scenery.backends.Shaders
import graphics.scenery.backends.vulkan.VulkanRenderer
import graphics.scenery.backends.vulkan.VulkanTexture
import graphics.scenery.compute.ComputeMetadata
import graphics.scenery.compute.InvocationType
import graphics.scenery.controls.TrackedStereoGlasses
import graphics.scenery.textures.Texture
import graphics.scenery.utils.Image
import graphics.scenery.utils.SystemHelpers
import graphics.scenery.utils.extensions.minus
import graphics.scenery.utils.extensions.plus
import graphics.scenery.utils.extensions.times
import graphics.scenery.volumes.*
import graphics.scenery.volumes.vdi.VDIBufferSizes
import graphics.scenery.volumes.vdi.VDIData
import graphics.scenery.volumes.vdi.VDIDataIO
import graphics.scenery.volumes.vdi.VDIMetadata
import net.imglib2.type.numeric.integer.UnsignedByteType
import net.imglib2.type.numeric.integer.UnsignedIntType
import net.imglib2.type.numeric.integer.UnsignedShortType
import net.imglib2.type.numeric.real.FloatType
import org.joml.*
import org.lwjgl.system.MemoryUtil
import tpietzsch.shadergen.generate.SegmentTemplate
import tpietzsch.shadergen.generate.SegmentType
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.lang.Math
import java.nio.ByteBuffer
import java.nio.file.Files
import java.nio.file.Path
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.atomic.AtomicInteger
import kotlin.concurrent.thread
import kotlin.streams.toList

class CompositorNode : RichNode() {
    @ShaderProperty
    var ProjectionOriginal = Matrix4f()

    @ShaderProperty
    var invProjectionOriginal = Matrix4f()

    @ShaderProperty
    var ViewOriginal = Matrix4f()

    @ShaderProperty
    var invViewOriginal = Matrix4f()

    @ShaderProperty
    var nw = 0f

    @ShaderProperty
    var doComposite = false

    @ShaderProperty
    var numProcesses = 0
}

class DistributedVolumes: SceneryBase("DistributedVolumeRenderer", windowWidth = 1280, windowHeight = 720, wantREPL = false) {

    private val vulkanProjectionFix =
        Matrix4f(
            1.0f,  0.0f, 0.0f, 0.0f,
            0.0f, -1.0f, 0.0f, 0.0f,
            0.0f,  0.0f, 0.5f, 0.0f,
            0.0f,  0.0f, 0.5f, 1.0f)

    fun Matrix4f.applyVulkanCoordinateSystem(): Matrix4f {
        val m = Matrix4f(vulkanProjectionFix)
        m.mul(this)

        return m
    }

    var volumes: HashMap<Int, BufferedVolume?> = java.util.HashMap()

    var hmd: TrackedStereoGlasses? = null

    lateinit var volumeManager: VolumeManager
    val compositor = CompositorNode()

    val generateVDIs = true
    val separateDepth = true
    val colors32bit = true
    val saveFinal = true
    val benchmarking = false
    var cnt_distr = 0
    var cnt_sub = 0
    var vdisGathered = 0
    val cam: Camera = DetachedHeadCamera(hmd)
    var camTarget = Vector3f(0f)

    val maxSupersegments = 20
    var maxOutputSupersegments = 20
    var numLayers = 0

    var commSize = 1
    var rank = 0
    var nodeRank = 0
    var pixelToWorld = 0.001f
    var volumeDims = Vector3f(0f)
    var dataset = ""
    var isCluster = false
    var basePath = ""
    var rendererConfigured = false

    var mpiPointer = 0L
    var allToAllColorPointer = 0L
    var allToAllDepthPointer = 0L
    var gatherColorPointer = 0L
    var gatherDepthPointer = 0L

    var volumesCreated = false
    var volumeManagerInitialized = false

    var vdisGenerated = AtomicInteger(0)
    var vdisDistributed = AtomicInteger(0)
    var vdisComposited = AtomicInteger(0)

    @Volatile
    var runGeneration = true
    @Volatile
    var runCompositing = false

    val singleGPUBenchmarks = false
    val colorMap = Colormap.get("hot")

    var dims = Vector3i(0)

    private external fun distributeVDIs(subVDIColor: ByteBuffer, subVDIDepth: ByteBuffer, sizePerProcess: Int, commSize: Int,
        colPointer: Long, depthPointer: Long, mpiPointer: Long)
    private external fun gatherCompositedVDIs(compositedVDIColor: ByteBuffer, compositedVDIDepth: ByteBuffer, compositedVDILen: Int, root: Int, myRank: Int, commSize: Int,
        colPointer: Long, depthPointer: Long, mpiPointer: Long)

    @Suppress("unused")
    fun setVolumeDims(dims: IntArray) {
        volumeDims = Vector3f(dims[0].toFloat(), dims[1].toFloat(), dims[2].toFloat())
    }

    @Suppress("unused")
    fun addVolume(volumeID: Int, dimensions: IntArray, pos: FloatArray, is16bit: Boolean) {
        logger.info("Trying to add the volume")
        logger.info("id: $volumeID, dims: ${dimensions[0]}, ${dimensions[1]}, ${dimensions[2]} pos: ${pos[0]}, ${pos[1]}, ${pos[2]}")

        while(!volumeManagerInitialized) {
            Thread.sleep(50)
        }

        val volume = if(is16bit) {
            Volume.fromBuffer(emptyList(), dimensions[0], dimensions[1], dimensions[2], UnsignedShortType(), hub)
        } else {
            Volume.fromBuffer(emptyList(), dimensions[0], dimensions[1], dimensions[2], UnsignedByteType(), hub)
        }
        volume.spatial().position = Vector3f(pos[0], pos[1], pos[2])

        volume.origin = Origin.FrontBottomLeft
        volume.spatial().needsUpdate = true
        volume.colormap = colorMap
        volume.pixelToWorldRatio = pixelToWorld
//        volume.pixelToWorldRatio = pixelToWorld

//        with(volume.transferFunction) {
//            this.addControlPoint(0.0f, 0.0f)
//            this.addControlPoint(0.2f, 0.1f)
//            this.addControlPoint(0.4f, 0.4f)
//            this.addControlPoint(0.8f, 0.6f)
//            this.addControlPoint(1.0f, 0.75f)
//        }

//        val tf = TransferFunction()
        val tf = TransferFunction.ramp(0.25f, 0.02f, 0.7f)

        with(tf) {
            if(dataset == "Stagbeetle" || dataset == "Stagbeetle_divided") {
                addControlPoint(0.0f, 0.0f)
                addControlPoint(0.005f, 0.0f)
                addControlPoint(0.01f, 0.3f)
            } else if (dataset == "Kingsnake") {
                addControlPoint(0.0f, 0.0f)
                addControlPoint(0.4f, 0.0f)
                addControlPoint(0.5f, 0.5f)
            } else if (dataset == "Beechnut") {
                addControlPoint(0.0f, 0.0f)
                addControlPoint(0.20f, 0.0f)
                addControlPoint(0.25f, 0.2f)
                addControlPoint(0.35f, 0.0f)
            } else if (dataset == "Simulation") {
                addControlPoint(0.0f, 0.0f)
                addControlPoint(0.2f, 0.0f)
                addControlPoint(0.4f, 0.0f)
                addControlPoint(0.45f, 0.1f)
                addControlPoint(0.5f, 0.10f)
                addControlPoint(0.55f, 0.1f)
                addControlPoint(0.83f, 0.1f)
                addControlPoint(0.86f, 0.4f)
                addControlPoint(0.88f, 0.7f)
//                addControlPoint(0.87f, 0.05f)
                addControlPoint(0.9f, 0.00f)
                addControlPoint(0.91f, 0.0f)
                addControlPoint(1.0f, 0.0f)
            } else if (dataset.contains("BonePlug")) {
                addControlPoint(0f, 0f)
                addControlPoint(0.01f, 0f)
                addControlPoint(0.11f, 0.01f)
            } else if (dataset == "Rotstrat") {
                logger.info("Using rotstrat transfer function")
                TransferFunction.ramp(0.0025f, 0.005f, 0.7f)
            }
            else {
                logger.info("Using a standard transfer function. Value of dataset: $dataset")
                TransferFunction.ramp(0.1f, 0.5f)
            }
        }


        volume.transferFunction = tf

        if(dataset.contains("BonePlug")) {
            volume.converterSetups[0].setDisplayRange(200.0, 12500.0)
            volume.colormap = Colormap.get("viridis")
        }

        if(dataset == "Rotstrat") {
            volume.colormap = Colormap.get("jet")
            volume.converterSetups[0].setDisplayRange(25000.0, 50000.0)
        }

        scene.addChild(volume)

        volumes[volumeID] = volume


        volumesCreated = true
    }

    @Suppress("unused")
    fun updateVolume(volumeID: Int, buffer: ByteBuffer) {
        while(volumes[volumeID] == null) {
            Thread.sleep(50)
        }
        logger.info("Volume $volumeID has been updated")
        volumes[volumeID]?.addTimepoint("t", buffer)
        volumes[volumeID]?.goToLastTimepoint()
    }

    fun bufferFromPath(file: Path): ByteBuffer {

        val infoFile: Path
        val volumeFiles: List<Path>

        if(Files.isDirectory(file)) {
            volumeFiles = Files.list(file).filter { it.toString().endsWith(".raw") && Files.isRegularFile(it) && Files.isReadable(it) }.toList()
            infoFile = file.resolve("stacks.info")
        } else {
            volumeFiles = listOf(file)
            infoFile = file.resolveSibling("stacks.info")
        }

        val lines = Files.lines(infoFile).toList()

        logger.debug("reading stacks.info (${lines.joinToString()}) (${lines.size} lines)")
        val dimensions = Vector3i(lines.get(0).split(",").map { it.toInt() }.toIntArray())
        dims = dimensions
        logger.debug("setting dim to ${dimensions.x}/${dimensions.y}/${dimensions.z}")
        logger.debug("Got ${volumeFiles.size} volumes")


        val volumes = CopyOnWriteArrayList<BufferedVolume.Timepoint>()
        val v = volumeFiles.first()
        val id = v.fileName.toString()
        val buffer: ByteBuffer by lazy {

            logger.debug("Loading $id from disk")
            val buffer = ByteArray(1024 * 1024)
            val stream = FileInputStream(v.toFile())
            val imageData: ByteBuffer = MemoryUtil.memAlloc((2 * dimensions.x * dimensions.y * dimensions.z))

            logger.debug("${v.fileName}: Allocated ${imageData.capacity()} bytes for UINT16 image of $dimensions")

            val start = System.nanoTime()
            var bytesRead = stream.read(buffer, 0, buffer.size)
            while (bytesRead > -1) {
                imageData.put(buffer, 0, bytesRead)
                bytesRead = stream.read(buffer, 0, buffer.size)
            }
            val duration = (System.nanoTime() - start) / 10e5
            logger.debug("Reading took $duration ms")

            imageData.flip()
            imageData
        }
        return buffer
    }

    fun setupVolumeManager() {
        val raycastShader: String
        val accumulateShader: String
        val compositeShader: String

        if(generateVDIs) {
            raycastShader = "VDIGenerator.comp"
            accumulateShader = "AccumulateVDI.comp"
            compositeShader = "VDICompositor.comp"
            numLayers = if(separateDepth) {
                1
            } else {
                3         // VDI supersegments require both front and back depth values, along with color
            }

            volumeManager = VolumeManager(
                hub, useCompute = true, customSegments = hashMapOf(
                    SegmentType.FragmentShader to SegmentTemplate(
                        this.javaClass,
                        raycastShader,
                        "intersectBoundingBox", "vis", "localNear", "localFar", "SampleVolume", "Convert", "Accumulate",
                    ),
                    SegmentType.Accumulator to SegmentTemplate(
//                                this.javaClass,
                        accumulateShader,
                        "vis", "localNear", "localFar", "sampleVolume", "convert",
                    ),
                ),
            )

            val colorBuffer = if(colors32bit) {
                MemoryUtil.memCalloc(windowHeight*windowWidth*4*maxSupersegments*numLayers * 4)
            } else {
                MemoryUtil.memCalloc(windowHeight*windowWidth*4*maxSupersegments*numLayers)
            }
            val depthBuffer = if(separateDepth) {
                MemoryUtil.memCalloc(windowHeight*windowWidth*4*maxSupersegments*2)
            } else {
                MemoryUtil.memCalloc(0)
            }

            val numGridCells = Vector3f(windowWidth.toFloat() / 8f, windowHeight.toFloat() / 8f, maxSupersegments.toFloat())
            val lowestLevel = MemoryUtil.memCalloc(numGridCells.x.toInt() * numGridCells.y.toInt() * numGridCells.z.toInt() * 4)

            val colorTexture: Texture
            val depthTexture: Texture
            val gridCells: Texture

            colorTexture = if(colors32bit) {
                Texture.fromImage(
                    Image(colorBuffer, numLayers * maxSupersegments, windowHeight, windowWidth), usage = hashSetOf(
                        Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture), type = FloatType(), channels = 4, mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)
            } else {
                Texture.fromImage(
                    Image(colorBuffer, numLayers * maxSupersegments, windowHeight, windowWidth), usage = hashSetOf(
                        Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
            }

            volumeManager.customTextures.add("OutputSubVDIColor")
            volumeManager.material().textures["OutputSubVDIColor"] = colorTexture

            if(separateDepth) {
                depthTexture = Texture.fromImage(
                    Image(depthBuffer, 2*maxSupersegments, windowHeight, windowWidth),  usage = hashSetOf(
                        Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture), type = FloatType(), channels = 1, mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)
                volumeManager.customTextures.add("OutputSubVDIDepth")
                volumeManager.material().textures["OutputSubVDIDepth"] = depthTexture
            }

            gridCells = Texture.fromImage(
                Image(lowestLevel, numGridCells.x.toInt(), numGridCells.y.toInt(), numGridCells.z.toInt()), channels = 1, type = UnsignedIntType(),
                usage = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
            volumeManager.customTextures.add("OctreeCells")
            volumeManager.material().textures["OctreeCells"] = gridCells
            volumeManager.customUniforms.add("doGeneration")
            volumeManager.shaderProperties["doGeneration"] = true

            hub.add(volumeManager)

            val compute = RichNode()
            compute.setMaterial(ShaderMaterial(Shaders.ShadersFromFiles(arrayOf("GridCellsToZero.comp"), this@DistributedVolumes::class.java)))

            compute.metadata["ComputeMetadata"] = ComputeMetadata(
                workSizes = Vector3i(numGridCells.x.toInt(), numGridCells.y.toInt(), 1),
                invocationType = InvocationType.Permanent
            )

            compute.material().textures["GridCells"] = gridCells

            compute.visible = false
//            scene.addChild(compute)

        } else {
            raycastShader = "VDIGenerator.comp"
            accumulateShader = "AccumulateVDI.comp"
            compositeShader = "AlphaCompositor.comp"
            volumeManager = VolumeManager(hub,
                useCompute = true,
                customSegments = hashMapOf(
                    SegmentType.FragmentShader to SegmentTemplate(
                        this.javaClass,
                        "ComputeRaycast.comp",
                        "intersectBoundingBox", "vis", "localNear", "localFar", "SampleVolume", "Convert", "Accumulate"),
                ))
            volumeManager.customTextures.add("OutputRender") //TODO: attach depth texture required for compositing
            volumeManager.shaderProperties["doGeneration"] = true

            val outputBuffer = MemoryUtil.memCalloc(windowWidth*windowHeight*4)
            val outputTexture = Texture.fromImage(
                Image(outputBuffer, windowWidth, windowHeight), usage = hashSetOf(
                    Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
            volumeManager.material().textures["OutputRender"] = outputTexture

            hub.add(volumeManager)

            val plane = FullscreenObject()
            scene.addChild(plane)
            plane.material().textures["diffuse"] = volumeManager.material().textures["OutputRender"]!!
        }

        compositor.name = "compositor node"
        compositor.setMaterial(ShaderMaterial(Shaders.ShadersFromFiles(arrayOf(compositeShader), this@DistributedVolumes::class.java)))
        if(generateVDIs) {
            val outputColours = MemoryUtil.memCalloc(maxOutputSupersegments*windowHeight*windowWidth*4*4 / commSize)
            val outputDepths = MemoryUtil.memCalloc(maxOutputSupersegments*windowHeight*windowWidth*4*2 / commSize)
            val compositedVDIColor = Texture.fromImage(Image(outputColours, maxOutputSupersegments, windowHeight,  windowWidth/commSize), channels = 4, usage = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture),
                type = FloatType(), mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)
            val compositedVDIDepth = Texture.fromImage(Image(outputDepths, 2 * maxOutputSupersegments, windowHeight,  windowWidth/commSize), channels = 1, usage = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture)
                , type = FloatType(), mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)
            compositor.material().textures["CompositedVDIColor"] = compositedVDIColor
            compositor.material().textures["CompositedVDIDepth"] = compositedVDIDepth
        } else {
            val outputColours = MemoryUtil.memCalloc(windowHeight*windowWidth*4 / commSize)
            val alphaComposited = Texture.fromImage(Image(outputColours, windowHeight,  windowWidth/commSize), usage = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
            compositor.material().textures["AlphaComposited"] = alphaComposited
        }
        compositor.metadata["ComputeMetadata"] = ComputeMetadata(
            workSizes = Vector3i(windowWidth/commSize, windowHeight, 1)
        )
        if(!singleGPUBenchmarks) {
            compositor.visible = true
            scene.addChild(compositor)
        } else {
            compositor.visible = false
        }
    }

    override fun init() {

        logger.info("setting renderer device id to: $nodeRank")
        System.setProperty("scenery.Renderer.DeviceId", nodeRank.toString())

        renderer = hub.add(Renderer.createRenderer(hub, applicationName, scene, windowWidth, windowHeight))

        setupVolumeManager()
        volumeManagerInitialized = true

        with(cam) {
            spatial {
//                position = Vector3f(3.174E+0f, -1.326E+0f, -2.554E+0f)
//                rotation = Quaternionf(-1.276E-2,  9.791E-1,  6.503E-2, -1.921E-1)

                position = Vector3f( 4.622E+0f, -9.060E-1f, -1.047E+0f) //V1 for kingsnake
                rotation = Quaternionf( 5.288E-2, -9.096E-1, -1.222E-1,  3.936E-1)
//
//                position = Vector3f(-2.607E+0f, -5.973E-1f,  2.415E+0f) // V1 for Beechnut
//                rotation = Quaternionf(-9.418E-2, -7.363E-1, -1.048E-1, -6.618E-1)

                position = Vector3f(4.908E+0f, -4.931E-1f, -2.563E+0f) //V1 for Simulation
                rotation = Quaternionf( 3.887E-2, -9.470E-1, -1.255E-1,  2.931E-1)

//                position = Vector3f( 1.897E+0f, -5.994E-1f, -1.899E+0f) //V1 for Boneplug
//                rotation = Quaternionf( 5.867E-5,  9.998E-1,  1.919E-2,  4.404E-3)
//
//                position = Vector3f( 3.183E+0f, -5.973E-1f, -1.475E+0f) //V2 for Beechnut
//                rotation = Quaternionf( 1.974E-2, -9.803E-1, -1.395E-1,  1.386E-1)

//                position = Vector3f( 4.458E+0f, -9.057E-1f,  4.193E+0f) //V2 for Kingsnake
//                rotation = Quaternionf( 1.238E-1, -3.649E-1,-4.902E-2,  9.215E-1)

//                position = Vector3f( 6.284E+0f, -4.932E-1f,  4.787E+0f) //V2 for Simulation
//                rotation = Quaternionf( 1.162E-1, -4.624E-1, -6.126E-2,  8.769E-1)
            }
            perspectiveCamera(50.0f, windowWidth, windowHeight)
            cam.farPlaneDistance = 20.0f

            scene.addChild(this)
        }

        val lights = (0 until 3).map {
            PointLight(radius = 15.0f)
        }

        lights.mapIndexed { i, light ->
            light.spatial().position = Vector3f(2.0f * i - 4.0f,  i - 1.0f, 0.0f)
            light.emissionColor = Vector3f(1.0f, 1.0f, 1.0f)
            light.intensity = 0.5f
            scene.addChild(light)
        }

        while(!rendererConfigured) {
            Thread.sleep(50)
        }

        logger.info("Exiting init function!")

        basePath = if(isCluster) {
            "/scratch/ws/1/anbr392b-test-workspace/argupta-vdi_generation/vdi_dumps/"
        } else {
            "/home/aryaman/TestingData/"
        }

        thread {
            if(generateVDIs) {
                if(singleGPUBenchmarks) {
                    doBenchmarks()
                } else {
                    manageVDIGeneration()
                }

            } else {
                saveScreenshots()
            }
        }
    }

    fun doBenchmarks() {
        while(renderer?.firstImageReady == false) {
            Thread.sleep(50)
        }

        while(!rendererConfigured) {
            Thread.sleep(50)
        }

        val pivot = Box(Vector3f(20.0f))
        pivot.material().diffuse = Vector3f(0.0f, 1.0f, 0.0f)
        pivot.spatial().position = Vector3f(volumeDims.x/2.0f, volumeDims.y/2.0f, volumeDims.z/2.0f)
//        parent.children.first().addChild(pivot)
        volumes[0]?.addChild(pivot)
//        parent.spatial().updateWorld(true)
        volumes[0]?.spatial()?.updateWorld(true)

        cam.target = pivot.spatial().worldPosition(Vector3f(0.0f))
        camTarget = pivot.spatial().worldPosition(Vector3f(0.0f))

        pivot.visible = false

        logger.info("Setting target to: ${pivot.spatial().worldPosition(Vector3f(0.0f))}")

        val model = volumes[0]?.spatial()?.world

        val vdiData = VDIData(
            VDIBufferSizes(),
            VDIMetadata(
                projection = cam.spatial().projection,
                view = cam.spatial().getTransformation(),
                volumeDimensions = volumeDims,
                model = model!!,
                nw = volumes[0]?.volumeManager?.shaderProperties?.get("nw") as Float,
                windowDimensions = Vector2i(cam.width, cam.height)
            )
        )

        var subVDIDepthBuffer: ByteBuffer? = null
        var subVDIColorBuffer: ByteBuffer?

        var numGenerated = 0

        val subVDIColor = volumeManager.material().textures["OutputSubVDIColor"]!!
        val colorCnt = AtomicInteger(0)

        (renderer as? VulkanRenderer)?.persistentTextureRequests?.add (subVDIColor to colorCnt)

        val depthCnt = AtomicInteger(0)
        var subVDIDepth: Texture? = null

        if(separateDepth) {
            subVDIDepth = volumeManager.material().textures["OutputSubVDIDepth"]!!
            (renderer as? VulkanRenderer)?.persistentTextureRequests?.add (subVDIDepth to depthCnt)
        }

        (renderer as VulkanRenderer).postRenderLambdas.add {
            vdiData.metadata.projection = cam.spatial().projection
            vdiData.metadata.view = cam.spatial().getTransformation()

            numGenerated++

            if(numGenerated == 10) {
                stats.clear("Renderer.fps")
            }

            if(numGenerated > 10) {
                rotateCamera(5f)
            }

            if(numGenerated > 155) {
//            if(numGenerated > 75) {
                //Stop
                val fps = stats.get("Renderer.fps")!!
                File("${dataset}_${windowWidth}_${windowHeight}_$maxSupersegments.csv").writeText("${fps.avg()};${fps.min()};${fps.max()};${fps.stddev()};${fps.data.size}")
                renderer?.shouldClose = true
            }

//            if(numGenerated % 20 == 0) {
//
//                subVDIColorBuffer = subVDIColor.contents
//                if (subVDIDepth != null) {
//                    subVDIDepthBuffer = subVDIDepth.contents
//                }
//
//                val fileName = "${dataset}VDI${numGenerated}_ndc"
//                SystemHelpers.dumpToFile(subVDIColorBuffer!!, "${fileName}_col")
//                SystemHelpers.dumpToFile(subVDIDepthBuffer!!, "${fileName}_depth")
//
//                val file = FileOutputStream(File("${dataset}vdidump$numGenerated"))
//                VDIDataIO.write(vdiData, file)
//                logger.info("written the dump")
//                file.close()
//            }
        }

    }

    private fun rotateCamera(degrees: Float) {
        cam.targeted = true
        val frameYaw = degrees / 180.0f * Math.PI.toFloat()
        val framePitch = 0f

        // first calculate the total rotation quaternion to be applied to the camera
        val yawQ = Quaternionf().rotateXYZ(0.0f, frameYaw, 0.0f).normalize()
        val pitchQ = Quaternionf().rotateXYZ(framePitch, 0.0f, 0.0f).normalize()

//        logger.info("Applying the rotation! camtarget is: $camTarget")

        val distance = (camTarget - cam.spatial().position).length()
        cam.spatial().rotation = pitchQ.mul(cam.spatial().rotation).mul(yawQ).normalize()
        cam.spatial().position = camTarget + cam.forward * distance * (-1.0f)
    }

    private fun saveScreenshots() {
        val r = (hub.get(SceneryElement.Renderer) as Renderer)

        while(!r.firstImageReady) {
            Thread.sleep(200)
        }

        val numScreenshots = 5

        dataset += "_${commSize}_${rank}"

        for(i in 1..numScreenshots) {
            val path = basePath + "${dataset}Screenshot$i"

            r.screenshot("$path.png")
            Thread.sleep(5000L)
        }
    }


    @Suppress("unused")
    fun stopRendering() {
        renderer?.shouldClose = true
    }

    fun fetchTexture(texture: Texture) : Int {
        val ref = VulkanTexture.getReference(texture)
        val buffer = texture.contents ?: return -1

        if(ref != null) {
            val start = System.nanoTime()
//            texture.contents = ref.copyTo(buffer, true)
            ref.copyTo(buffer, false)
            val end = System.nanoTime()
//            logger.info("The request textures of size ${texture.contents?.remaining()?.toFloat()?.div((1024f*1024f))} took: ${(end.toDouble()-start.toDouble())/1000000.0}")
        } else {
            logger.error("In fetchTexture: Texture not accessible")
        }

        return 0
    }

    private fun manageVDIGeneration() {

        while(renderer?.firstImageReady == false) {
            Thread.sleep(50)
        }

        while(!rendererConfigured) {
            Thread.sleep(50)
        }

        basePath = if(isCluster) {
            "/scratch/ws/1/anbr392b-test-workspace/argupta-vdi_generation/vdi_dumps/"
        } else {
            "/home/aryaman/TestingData/"
        }

//        camTarget = Vector3f(1.920E+0f, -1.920E+0f,  1.800E+0f)
//        camTarget = Vector3f(1.920E+0f, -6.986E-1f,  6.855E-1f) //BonePlug
//        camTarget = Vector3f(1.920E+0f, -6.986E-1f,  6.855E-1f) //BonePlug
        camTarget = Vector3f(1.920E+0f, -1.920E+0f,  1.800E+0f) //Rotstrat

         val model = volumes[0]?.spatial()?.world

        val vdiData = VDIData(
            VDIBufferSizes(),
            VDIMetadata(
                projection = cam.spatial().projection,
                view = cam.spatial().getTransformation(),
                volumeDimensions = volumeDims,
                model = model!!,
                nw = volumes[0]?.volumeManager?.shaderProperties?.get("nw") as Float,
                windowDimensions = Vector2i(cam.width, cam.height)
            )
        )

        compositor.nw = vdiData.metadata.nw
        compositor.ViewOriginal = vdiData.metadata.view
        compositor.invViewOriginal = Matrix4f(vdiData.metadata.view).invert()
        compositor.ProjectionOriginal = Matrix4f(vdiData.metadata.projection).applyVulkanCoordinateSystem()
        compositor.invProjectionOriginal = Matrix4f(vdiData.metadata.projection).applyVulkanCoordinateSystem().invert()
        compositor.numProcesses = commSize

        var colorTexture = volumeManager.material().textures["OutputSubVDIColor"]!!
        var depthTexture = volumeManager.material().textures["OutputSubVDIDepth"]!!

        var compositedColor =
            if(generateVDIs) {
                compositor.material().textures["CompositedVDIColor"]!!
            } else {
                compositor.material().textures["AlphaComposited"]!!
            }
        var compositedDepth = compositor.material().textures["CompositedVDIDepth"]!!

        (renderer as VulkanRenderer).postRenderLambdas.add {
            if(runGeneration) {

                colorTexture = volumeManager.material().textures["OutputSubVDIColor"]!!
                depthTexture = volumeManager.material().textures["OutputSubVDIDepth"]!!

                val col = fetchTexture(colorTexture)
                val depth = if(separateDepth) {
                    fetchTexture(depthTexture)
                } else {
                    0
                }

                if(col < 0) {
                    logger.error("Error fetching the color subVDI!!")
                }
                if(depth < 0) {
                    logger.error("Error fetching the depth subVDI!!")
                }

                vdisGenerated.incrementAndGet()
                runGeneration = false
            }
            if(runCompositing) {
                compositedColor = compositor.material().textures["CompositedVDIColor"]!!
                compositedDepth = compositor.material().textures["CompositedVDIDepth"]!!

                val col = fetchTexture(compositedColor)
                val depth = if(separateDepth) {
                    fetchTexture(compositedDepth)
                } else {
                    0
                }

                if(col < 0) {
                    logger.error("Error fetching the color compositedVDI!!")
                }
                if(depth < 0) {
                    logger.error("Error fetching the depth compositedVDI!!")
                }

                vdisComposited.incrementAndGet()
                runCompositing = false
                rotateCamera(10f)
                vdiData.metadata.projection = cam.spatial().projection
                vdiData.metadata.view = cam.spatial().getTransformation()
                runGeneration = true
            }

            if(vdisDistributed.get() > vdisComposited.get()) {
                runCompositing = true
            }
        }

        (renderer as VulkanRenderer).postRenderLambdas.add {
            if(runCompositing) {
//                logger.info("SETTING DO_COMPOSITE TO TRUE!")
            }
            compositor.doComposite = runCompositing
            volumes[0]?.volumeManager?.shaderProperties?.set("doGeneration", runGeneration)
        }

        var generatedSoFar = 0
        var compositedSoFar = 0

        dataset += "_${commSize}_${rank}"

        var start = 0L
        var end = 0L

        var start_complete = 0L
        var end_complete = 0L

        while(true) {

            start_complete = System.nanoTime()

            var subVDIDepthBuffer: ByteBuffer? = null
            var subVDIColorBuffer: ByteBuffer?
            var bufferToSend: ByteBuffer? = null

            var compositedVDIDepthBuffer: ByteBuffer?
            var compositedVDIColorBuffer: ByteBuffer?

            start = System.nanoTime()
            while((vdisGenerated.get() <= generatedSoFar)) {
                Thread.sleep(5)
            }
            end = System.nanoTime() - start

//            logger.info("Waiting for VDI generation took: ${end/1e9}")


//            logger.warn("C1: vdis generated so far: $generatedSoFar and the new value of vdisgenerated: ${vdisGenerated.get()}")

            generatedSoFar = vdisGenerated.get()

            subVDIColorBuffer = colorTexture.contents
            if(separateDepth) {
                subVDIDepthBuffer = depthTexture!!.contents
            }
            compositedVDIColorBuffer = compositedColor.contents
            compositedVDIDepthBuffer = compositedDepth.contents

//
//            compositor.ViewOriginal = vdiData.metadata.view
//            compositor.invViewOriginal = Matrix4f(vdiData.metadata.view).invert()
//            compositor.ProjectionOriginal = Matrix4f(vdiData.metadata.projection).applyVulkanCoordinateSystem()
//            compositor.invProjectionOriginal = Matrix4f(vdiData.metadata.projection).applyVulkanCoordinateSystem().invert()

            if(!benchmarking) {
                logger.info("Dumping sub VDI files")
                SystemHelpers.dumpToFile(subVDIColorBuffer!!, basePath + "${dataset}SubVDI${cnt_sub}_ndc_col")
                SystemHelpers.dumpToFile(subVDIDepthBuffer!!, basePath + "${dataset}SubVDI${cnt_sub}_ndc_depth")
                logger.info("File dumped")
            }

            if(subVDIColorBuffer == null || subVDIDepthBuffer == null) {
                logger.info("CALLING DISTRIBUTE EVEN THOUGH THE BUFFERS ARE NULL!!")
            }

//            logger.info("For distributed the color buffer isdirect: ${subVDIColorBuffer!!.isDirect()} and depthbuffer: ${subVDIDepthBuffer!!.isDirect()}")

            start = System.nanoTime()
            distributeVDIs(subVDIColorBuffer!!, subVDIDepthBuffer!!, windowHeight * windowWidth * maxSupersegments * 4 / commSize, commSize, allToAllColorPointer,
            allToAllDepthPointer, mpiPointer)
            end = System.nanoTime() - start

//            logger.info("Distributing VDIs took: ${end/1e9}")


//            logger.info("Back in the management function")

            if(!benchmarking) {
//                Thread.sleep(1000)
//
//                subVDIColorBuffer.clear()
//                subVDIDepthBuffer.clear()
            }

            start = System.nanoTime()
            while(vdisComposited.get() <= compositedSoFar) {
                Thread.sleep(5)
            }
            end = System.nanoTime() - start

//            logger.info("Waiting for composited generation took: ${end/1e9}")

            compositedSoFar = vdisComposited.get()

            //fetch the composited VDI

            if(compositedVDIColorBuffer == null || compositedVDIDepthBuffer == null) {
                logger.info("CALLING GATHER EVEN THOUGH THE BUFFER(S) ARE NULL!!")
            }

            if(!benchmarking) {
                logger.info("Dumping sub VDI files")
                SystemHelpers.dumpToFile(compositedVDIColorBuffer!!, basePath + "${dataset}CompositedVDI${cnt_sub}_ndc_col")
                SystemHelpers.dumpToFile(compositedVDIDepthBuffer!!, basePath + "${dataset}CompositedVDI${cnt_sub}_ndc_depth")
                logger.info("File dumped")
                cnt_sub++
            }

//            logger.info("For gather the color buffer isdirect: ${compositedVDIColorBuffer!!.isDirect()} and depthbuffer: ${compositedVDIDepthBuffer!!.isDirect()}")

            start = System.nanoTime()
            gatherCompositedVDIs(compositedVDIColorBuffer!!, compositedVDIDepthBuffer!!, windowHeight * windowWidth * maxOutputSupersegments * 4 * numLayers/ commSize, 0,
                rank, commSize, gatherColorPointer, gatherDepthPointer, mpiPointer) //3 * commSize because the supersegments here contain only 1 element
            end = System.nanoTime() - start

//            logger.info("Gather took: ${end/1e9}")


            if(saveFinal /*&& (rank == 0)*/) {
                val file = FileOutputStream(File(basePath + "${dataset}vdi_${windowWidth}_${windowHeight}_${maxSupersegments}_0_dump$vdisGathered"))
                VDIDataIO.write(vdiData, file)
                logger.info("written the dump $vdisGathered")
                file.close()
            }

            vdisGathered++

            if(!benchmarking) {
//                Thread.sleep(1000)
//
//                compositedVDIColorBuffer.clear()
//                compositedVDIDepthBuffer.clear()
//
                logger.info("Back in the management function after gathering and streaming")
//                Thread.sleep(1000)
            }

            end_complete = System.nanoTime() - start_complete

//            logger.info("Whole iteration took: ${end_complete/1e9}")
        }
    }

    override fun inputSetup() {
        setupCameraModeSwitching()

//        inputHandler?.addBehaviour("rotate_camera", ClickBehaviour { _, _ ->
//            addVolume(0, intArrayOf(200, 200, 200), floatArrayOf(0f, 0f, -3.5f))
//        })
//        inputHandler?.addKeyBinding("rotate_camera", "R")
    }

    @Suppress("unused")
    fun uploadForCompositing(vdiSetColour: ByteBuffer, vdiSetDepth: ByteBuffer) {
        //Receive the VDIs and composite them

//        logger.info("In the composite function")

//        if(saveFiles) {
        val model = volumes[0]?.spatial()?.world

//        val vdiData = VDIData(
//            VDIMetadata(
//                projection = cam.spatial().projection,
//                view = cam.spatial().getTransformation(),
//                volumeDimensions = volumeDims,
//                model = model!!,
//                nw = volumes[0]?.volumeManager?.shaderProperties?.get("nw") as Float,
//                windowDimensions = Vector2i(cam.width, cam.height)
//            )
//        )
//
//        val duration = measureNanoTime {
//            val file = FileOutputStream(File(basePath + "${dataset}vdidump$cnt_distr"))
////                    val comp = GZIPOutputStream(file, 65536)
//            VDIDataIO.write(vdiData, file)
//            logger.info("written the dump")
//            file.close()
//        }

        if(!benchmarking) {
            logger.info("Dumping to file in the composite function")
            SystemHelpers.dumpToFile(vdiSetColour, basePath + "${dataset}SetOfVDI${cnt_distr}_ndc_col")
            SystemHelpers.dumpToFile(vdiSetDepth, basePath + "${dataset}SetOfVDI${cnt_distr}_ndc_depth")
            logger.info("File dumped")
            cnt_distr++
        }
//        }

        if(generateVDIs) {
            compositor.material().textures["VDIsColor"] = Texture(Vector3i(maxSupersegments * numLayers, windowHeight, windowWidth), 4, contents = vdiSetColour, usageType = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture),
                    type = FloatType(), mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)

            if(separateDepth) {
                compositor.material().textures["VDIsDepth"] = Texture(Vector3i(2 * maxSupersegments, windowHeight, windowWidth), 1, contents = vdiSetDepth, usageType = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture),
                    type = FloatType(), mipmap = false, normalized = false, minFilter = Texture.FilteringMode.NearestNeighbour, maxFilter = Texture.FilteringMode.NearestNeighbour)
            }
        } else {
            compositor.material().textures["VDIsColor"] = Texture(Vector3i(windowHeight, windowWidth, 1), 4, contents = vdiSetColour, usageType = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
            compositor.material().textures["VDIsDepth"] = Texture(Vector3i(windowHeight, windowWidth, 1), 4, contents = vdiSetDepth, usageType = hashSetOf(Texture.UsageType.LoadStoreImage, Texture.UsageType.Texture))
        }
//        logger.info("Updated the textures to be composited")

        vdisDistributed.incrementAndGet()

//        compute.visible = true
    }


    companion object {
        @JvmStatic
        fun main(args: Array<String>) {
            DistributedVolumes().main()
        }
    }
}