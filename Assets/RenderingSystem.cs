using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using Unity.Mathematics;
using UnityEngine.Profiling;

public class RenderingSystem : MonoBehaviour
{
	public Mesh testMesh;

	public Texture2D rt1;
	
	public bool jobifyVertex = true;
	public bool jobifyPixels = true;

	protected int _width, _height;

	float4x4 model;
	float4x4 view;
	float4x4 proj;

	NativeArray<VertexIn> mesh;
	NativeArray<int> triangles;
	
	NativeArray<float> depthBuffer;
	NativeArray<Color32> colorBuffer;

	NativeArray<VertexOut> vertexOutput;

	NativeArray<float> clearDepthBuffer;
	NativeArray<Color32> clearColor;

	public struct VertexIn
	{
		public float4 position;
		public float4 normal;
		public float4 texcoord0;
	}
	
	public struct VertexOut
	{
		public float4 position;
		public float4 normal;
		public float4 texcoord0;
	}
	
	// Use this for initialization
	void Start ()
	{
		InitData();
		UpdateTexture();

		model = float4x4.identity;
		view = float4x4.lookAt(new float3(0, 1, -7), new float3(0,0,1), new float3(0,-1,0));
		proj = Matrix4x4.Perspective(60.0f, _width / (float)_height, 0.1f, 30.0f);
	}

	void InitData()
	{
		mesh = new NativeArray<VertexIn>(testMesh.vertices.Length, Allocator.Persistent);
		for (int i = 0; i < mesh.Length; ++i)
		{
			VertexIn value;
			value.position = new float4(testMesh.vertices[i], 1.0f);
			
			value.normal = new float4(testMesh.normals[i], 0.0f);
			value.texcoord0 = new float4(testMesh.uv[i], 0.0f, 0.0f);

			mesh[i] = value;
		}

		
		triangles = new NativeArray<int>(testMesh.triangles, Allocator.Persistent);
		vertexOutput = new NativeArray<VertexOut>(triangles.Length, Allocator.Persistent);
	}
	
	void UpdateTexture()
	{
		_width = Screen.width;
		_height = Screen.height;

		if(depthBuffer.IsCreated)
			depthBuffer.Dispose();
		
		depthBuffer = new NativeArray<float>(_width * _height, Allocator.Persistent);
		
		rt1 = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
		
		if(clearDepthBuffer.IsCreated)
			clearDepthBuffer.Dispose();
		
		if(clearColor.IsCreated)
			clearColor.Dispose();
		
		clearDepthBuffer = new NativeArray<float>(_width * _height, Allocator.Persistent);
		clearColor = new NativeArray<Color32>(_width * _height, Allocator.Persistent);
		for (int i = 0; i < _width * _height; ++i)
		{
			clearDepthBuffer[i] = 1.0f;
			clearColor[i] = new Color32(0,0,0,255);
		}
		
		DisplayResult.Instance.rt = rt1;
	}

	void OnDestroy()
	{
		triangles.Dispose();
		mesh.Dispose();
		vertexOutput.Dispose();
		
		depthBuffer.Dispose();
		
		clearDepthBuffer.Dispose();
		clearColor.Dispose();
	}

	float angle = 0;
	
	// Update is called once per frame
	void Update () 
	{
		if (Screen.width != _width || Screen.height != _height)
		{
			//UpdateTexture();
		}

		angle += Time.deltaTime * Mathf.PI * 0.5f;

		if (angle > Math.PI * 2) angle -= Mathf.PI * 2;
		
		model = float4x4.eulerXYZ(0, angle, 0);
		Draw();
	}
	
	[BurstCompile(CompileSynchronously = true)]
	public struct VertexProcessingJob : IJobParallelFor
	{
		[ReadOnly]
		public float4x4 model, view, proj;
		[ReadOnly]
		public NativeArray<int> triangles;
		[ReadOnly]
		public NativeArray<VertexIn> input;
		
		[WriteOnly]
		public NativeArray<VertexOut> output;
    
		public void Execute(int i)
		{
			VertexIn inVal = input[triangles[i]];

			VertexOut outVal;

			outVal.position = math.mul(model, inVal.position);
			outVal.position = math.mul(view, outVal.position);
			outVal.position = math.mul(proj, outVal.position);

			outVal.texcoord0 = inVal.texcoord0;
			outVal.normal = math.mul(model, inVal.normal);

			output[i] = outVal;
		}
	}

	[BurstCompile(CompileSynchronously = true)]
	public struct RasterizePixelJob : IJobParallelFor
	{
		[ReadOnly]
		public int width, height;
		[ReadOnly]
		public NativeArray<VertexOut> input;

		[NativeDisableParallelForRestriction] public NativeArray<float> depthbuffer;
		[NativeDisableParallelForRestriction] public NativeArray<Color32> output;

		public void Execute(int i)
		{
			//TODO there must be a better way to jump in the input 3 by 3 ...
			if (i % 3 != 0)
			{
				return;
			}
						
			VertexOut v1 = input[i + 0];
			VertexOut v2 = input[i + 1];
			VertexOut v3 = input[i + 2];


			v1.position.xyz = v1.position.xyz / v1.position.w;
			v2.position.xyz = v2.position.xyz / v2.position.w;
			v3.position.xyz = v3.position.xyz / v3.position.w;


			v1.position.x = (v1.position.x + 0.5f) * width;
			v2.position.x = (v2.position.x + 0.5f) * width;
			v3.position.x = (v3.position.x + 0.5f) * width;

			v1.position.y =  height - (v1.position.y + 0.5f) * height;
			v2.position.y =  height - (v2.position.y + 0.5f) * height;
			v3.position.y =  height - (v3.position.y + 0.5f) * height;

			// 28.4 fixed-point coordinates
			int Y1 = (int)math.round(16.0f * v1.position.y);
			int Y2 = (int)math.round(16.0f * v2.position.y);
			int Y3 = (int)math.round(16.0f * v3.position.y);

			int X1 = (int)math.round(16.0f * v1.position.x);
			int X2 = (int)math.round(16.0f * v2.position.x);
			int X3 = (int)math.round(16.0f * v3.position.x);

			// Deltas
			int DX12 = X1 - X2;
			int DX23 = X2 - X3;
			int DX31 = X3 - X1;

			int DY12 = Y1 - Y2;
			int DY23 = Y2 - Y3;
			int DY31 = Y3 - Y1;

			// Fixed-point deltas
			int FDX12 = DX12 << 4;
			int FDX23 = DX23 << 4;
			int FDX31 = DX31 << 4;

			int FDY12 = DY12 << 4;
			int FDY23 = DY23 << 4;
			int FDY31 = DY31 << 4;

			// Bounding rectangle
			int minx = (Min3(X1, X2, X3) + 0xF) >> 4;
			int maxx = (Max3(X1, X2, X3) + 0xF) >> 4;
			int miny = (Min3(Y1, Y2, Y3) + 0xF) >> 4;
			int maxy = (Max3(Y1, Y2, Y3) + 0xF) >> 4;

			if (minx >= width || miny >= height || maxx < 0 || maxy < 0)
			{
				return; //rect outside of screen
			}

			minx = (minx < 0 ? 0 : minx);
			miny = (miny < 0 ? 0 : miny);
			maxx = (maxx >= width ? width - 1 : maxx);
			maxy = (maxy >= height ? height - 1 : maxy);

			// Block size, standard 8x8 (must be power of two)
			const int q = 32;

			// Start in corner of 8x8 block
			minx &= ~(q - 1);
			miny &= ~(q - 1);

			// Half-edge constants
			int C1 = DY12 * X1 - DX12 * Y1;
			int C2 = DY23 * X2 - DX23 * Y2;
			int C3 = DY31 * X3 - DX31 * Y3;

			// Correct for fill convention
			if (DY12 < 0 || (DY12 == 0 && DX12 > 0)) C1++;
			if (DY23 < 0 || (DY23 == 0 && DX23 > 0)) C2++;
			if (DY31 < 0 || (DY31 == 0 && DX31 > 0)) C3++;

			
			// Loop through blocks
			for (int y = miny; y < maxy; y += q)
			{
				for (int x = minx; x < maxx; x += q)
				{
					// Corners of block
					int x0 = x << 4;
					int x1 = (x + q - 1) << 4;
					int y0 = y << 4;
					int y1 = (y + q - 1) << 4;

					// Evaluate half-space functions
					int a00 = math.bool_to_int(C1 + DX12 * y0 - DY12 * x0 > 0);
					int a10 = math.bool_to_int(C1 + DX12 * y0 - DY12 * x1 > 0);
					int a01 = math.bool_to_int(C1 + DX12 * y1 - DY12 * x0 > 0);
					int a11 = math.bool_to_int(C1 + DX12 * y1 - DY12 * x1 > 0);
					int a = (a00 << 0) | (a10 << 1) | (a01 << 2) | (a11 << 3);

					int b00 = math.bool_to_int(C2 + DX23 * y0 - DY23 * x0 > 0);
					int b10 = math.bool_to_int(C2 + DX23 * y0 - DY23 * x1 > 0);
					int b01 = math.bool_to_int(C2 + DX23 * y1 - DY23 * x0 > 0);
					int b11 = math.bool_to_int(C2 + DX23 * y1 - DY23 * x1 > 0);
					int b = (b00 << 0) | (b10 << 1) | (b01 << 2) | (b11 << 3);

					int c00 = math.bool_to_int(C3 + DX31 * y0 - DY31 * x0 > 0);
					int c10 = math.bool_to_int(C3 + DX31 * y0 - DY31 * x1 > 0);
					int c01 = math.bool_to_int(C3 + DX31 * y1 - DY31 * x0 > 0);
					int c11 = math.bool_to_int(C3 + DX31 * y1 - DY31 * x1 > 0);
					int c = (c00 << 0) | (c10 << 1) | (c01 << 2) | (c11 << 3);

					// Skip block when outside an edge
					if (a == 0x0 || b == 0x0 || c == 0x0) continue;

					// Accept whole block when totally covered
					if (a == 0xF && b == 0xF && c == 0xF)
					{
						for (int iy = y; iy < y + q; iy++)
						{
							for (int ix = x; ix < x + q; ix++)
							{
								handleBlock(ix, iy, v1, v2, v3);
							}
						}
					}
					else
					{
						int CY1 = C1 + DX12 * y0 - DY12 * x0;
						int CY2 = C2 + DX23 * y0 - DY23 * x0;
						int CY3 = C3 + DX31 * y0 - DY31 * x0;

						for (int iy = y; iy < y + q; iy++)
						{
							int CX1 = CY1;
							int CX2 = CY2;
							int CX3 = CY3;

							for (int ix = x; ix < x + q; ix++)
							{
								if (CX1 > 0 && CX2 > 0 && CX3 > 0)
								{
									handleBlock(ix, iy, v1, v2, v3);
								}

								CX1 -= FDY12;
								CX2 -= FDY23;
								CX3 -= FDY31;
							}

							CY1 += FDX12;
							CY2 += FDX23;
							CY3 += FDX31;
						}
					}
				}	
			}
		}
		
		void handleBlock(int x, int y, VertexOut v1, VertexOut v2, VertexOut v3)
		{
			VertexOut v;
			float4 resultColor;
								
			float3 bar = Barycentric(v1.position,v2.position,v3.position, x, y);

			v = Interpolate(v1, v2, v3, bar);

			if (v.position.z < 0.0f || v.position.z > 1.0f || v.position.z > depthbuffer[y * width + x])
			{
				return;
			}
			

			Color32 outputColor = output[y * width + x];

			float lum = math.max(0.0f, math.dot(v.normal.xyz, new float3(-0.71f, -0.71f, 0)));

			resultColor.x = 0.3f + lum;
			resultColor.y = 0.3f + lum;
			resultColor.z = 0.3f + lum;
			resultColor.w = 0;

			resultColor = math.saturate(resultColor);

			outputColor.r = (byte)Mathf.FloorToInt(resultColor.x * 255);
			outputColor.g = (byte)Mathf.FloorToInt(resultColor.y * 255);
			outputColor.b = (byte)Mathf.FloorToInt(resultColor.z * 255);
			outputColor.a = (byte)Mathf.FloorToInt(resultColor.w * 255);

			output[y * width + x] = outputColor;
			depthbuffer[y * width + x] = v.position.z;
		}
	}

	public void Draw()
	{
		colorBuffer = rt1.GetRawTextureData<Color32>();

		colorBuffer.CopyFrom(clearColor);
		depthBuffer.CopyFrom(clearDepthBuffer);
				
		var vertexJob = new VertexProcessingJob();
		vertexJob.model = model;
		vertexJob.proj = proj;
		vertexJob.view = view;
		vertexJob.triangles = triangles;
		vertexJob.input = mesh;

		vertexJob.output = vertexOutput;
		
		if (jobifyVertex)
		{
			JobHandle handle = vertexJob.Schedule(vertexJob.output.Length, 18);

			handle.Complete();
		}
		else
		{
			for(int i = 0 ; i < vertexJob.output.Length; ++i)
				vertexJob.Execute(i);
		}

		RasterizePixelJob rasterizePixelJob = new RasterizePixelJob();
		rasterizePixelJob.depthbuffer = depthBuffer;
		rasterizePixelJob.output = colorBuffer;
		rasterizePixelJob.input = vertexOutput;
		rasterizePixelJob.width = _width;
		rasterizePixelJob.height = _height;
		
		if (jobifyPixels)
		{
			var handle = rasterizePixelJob.Schedule(vertexOutput.Length, 3);
			handle.Complete();
		}
		else
		{
			for(int i = 0; i < vertexOutput.Length; ++i)
				rasterizePixelJob.Execute(i);
		}

		rt1.Apply();
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	static int Min3(int a, int b, int c)
	{
		int t = (a > b ? b : a);

		return t > c ? c : t;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	static int Max3(int a, int b, int c)
	{
		int t = (a < b ? b : a);

		return t < c ? c : t;
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	static float3 Barycentric(float4 a, float4 b, float4 c, float x, float y)
	{
		float det = ((b.y-c.y)*(a.x-c.x) + (c.x-b.x)*(a.y-c.y));

		float xb = ((b.y - c.y)*(x - c.x) + (c.x - b.x)*(y - c.y)) / det;
		float yb = ((c.y - a.y)*(x - c.x) + (a.x - c.x)*(y - c.y)) / det;
		float zb = 1.0f - xb - yb;
		
		return new float3(xb,yb,zb);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	static float4 InterpolatedFromBarycentric(float4 v1, float4 v2, float4 v3, float3 barycentric)
	{
		return new float4(v1 * barycentric.x + v2 * barycentric.y + v3 * barycentric.z);
	}

	[MethodImpl(MethodImplOptions.AggressiveInlining)]
	static VertexOut Interpolate(VertexOut v1, VertexOut v2, VertexOut v3, float3 barycentric)
	{
		VertexOut output = new VertexOut();

		output.position = InterpolatedFromBarycentric(v1.position, v2.position, v3.position, barycentric);
		output.normal = InterpolatedFromBarycentric(v1.normal, v2.normal, v3.normal, barycentric);
		output.texcoord0 = InterpolatedFromBarycentric(v1.texcoord0, v2.texcoord0, v3.texcoord0, barycentric);

		return output;
	}
}
