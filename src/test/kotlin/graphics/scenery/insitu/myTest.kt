package graphics.scenery.insitu

import graphics.scenery.volumes.vdi.VDICompressor
import java.nio.ByteBuffer

fun main() {
    val buffer : ByteBuffer =  ByteBuffer.allocateDirect(64)
    //val compressor = VDICompressor()
    //val compressionTool = VDICompressor.CompressionTool.LZ4;

    //println(compressor.returnCompressBound(10000, compressionTool))
    buffer.position(0)
    for(i in 0 until 4){
        println(i)
        buffer.putInt(i)
    }
    buffer.putLong(1000)
    println(buffer.getLong(16))
}