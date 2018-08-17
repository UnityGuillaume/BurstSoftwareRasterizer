using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DisplayResult : MonoBehaviour
{
	static public DisplayResult Instance;

	public Texture2D rt;
	
	// Use this for initialization
	void Awake ()
	{
		Instance = this;
	}

	void OnRenderImage(RenderTexture src, RenderTexture dest)
	{
		if(rt != null)
			Graphics.Blit(rt, dest);
	}
}
