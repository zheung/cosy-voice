import os, sys

import argparse
import gradio as gr
import random

import numpy as np, torch, torchaudio, librosa

from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{ROOT_DIR}/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed



cosyvoice = None
datasEmpty = None

speakers_key = {}
speakers = []


valMax = 0.8
def postprocess(speech, top_db=60, hop_length=220, win_length=440):
	speech, _ = librosa.effects.trim(
		speech, top_db=top_db,
		frame_length=win_length,
		hop_length=hop_length
	)
	if speech.abs().max() > valMax:
		speech = speech / speech.abs().max() * valMax
	speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
	return speech



rateSamplePrompt = 16000
def createAudio(
	modeInference,
	textCreate, textInstruct, audioSourceCoverTimbre,
	idSpeaker,
	textPrompt, audioPromptUpload, audioPromptRecord,
	speed, stream, seed
):
	global speakers, speakers_key

	# if textInstruct == '':
	# 	yield (cosyvoice.sample_rate, datasEmpty)
	# 	return gr.Warning('您正在使用指令模式, 请输入指令文本')


	set_all_random_seed(seed)
	idSpeakerTemp = f'temp-{idSpeaker}'
	cosyvoice.frontend.spk2info[idSpeakerTemp] = dict(speakers_key[idSpeaker])

	textNow = datetime.now().strftime('%m%d-%H%M%S')

	if modeInference == '普通':
		logging.info(f'[普通合成]开始：音色[{idSpeaker}]，文本[{textCreate}]')
		for data in cosyvoice.inference_zero_shot(textCreate, '', '', idSpeakerTemp, stream, speed):
			logging.info(f'[普通合成]结束：音色[{idSpeaker}]，文本[{textCreate}]')

			yield (cosyvoice.sample_rate, data['tts_speech'].numpy().flatten())

	if modeInference == '指令':
		logging.info(f'[指令合成]开始：音色[{idSpeaker}]，指令[{textInstruct}]，文本[{textCreate}]')

		for data in cosyvoice.inference_instruct2(textCreate, textInstruct, '', idSpeakerTemp, stream, speed):
			logging.info(f'[指令合成]结束：音色[{idSpeaker}]，指令[{textInstruct}]，文本[{textCreate}]')

			yield (cosyvoice.sample_rate, data['tts_speech'].numpy().flatten())

	if modeInference == '跨语种':
		logging.info(f'[跨语种合成]开始：音色[{idSpeaker}]，文本[{textCreate}]')

		for data in cosyvoice.inference_cross_lingual(textCreate, '', idSpeakerTemp, stream, speed):
			logging.info(f'[跨语种合成]结束：音色[{idSpeaker}]，文本[{textCreate}]')

			yield (cosyvoice.sample_rate, data['tts_speech'].numpy().flatten())



def insertSpeaker(idSpeakerInsert, textPrompt, audioPromptUpload, audioPromptRecord):
	global speakers, speakers_key

	idSpeakerInsert = str.strip(idSpeakerInsert)
	if idSpeakerInsert is None or idSpeakerInsert == '':
		gr.Error('请输入音色ID')
		return

	if idSpeakerInsert in speakers_key:
		return gr.Warning('音色ID已存在')


	if audioPromptUpload is not None:
		audioPrompt = audioPromptUpload
	elif audioPromptRecord is not None:
		audioPrompt = audioPromptRecord
	else:
		return gr.Warning('请先上传或录制音频')


	audioSpeech = postprocess(load_wav(audioPrompt, rateSamplePrompt))


	speaker = cosyvoice.frontend.frontend_zero_shot('', textPrompt, audioSpeech, cosyvoice.sample_rate, '')
	del speaker['text']
	del speaker['text_len']

	speakers_key[idSpeakerInsert] = speaker

	torch.save(speakers_key, args.fileSpeakers)

	speakers = list(speakers_key.keys())

	logging.info(f'{idSpeakerInsert}音色已复刻')

	return [gr.update( value=None, choices=speakers), gr.update( value=None, choices=speakers), gr.update(interactive=True)]

def updateSpeaker(idsSpeakersOld, idSpeakerNew):
	global speakers, speakers_key

	if(len(idsSpeakersOld) == 0):
		return gr.Warning('请选择要修改ID的音色')
	if(len(idsSpeakersOld) > 1):
		return gr.Warning('不能同时修改多个音色的ID')

	speaker = speakers_key[idsSpeakersOld[0]]
	del speakers_key[idsSpeakersOld[0]]
	speakers_key[idSpeakerNew] = speaker

	speakers = list(speakers_key.keys())

	return [gr.update( value=None, choices=speakers), gr.update( value=None, choices=speakers), gr.update(value='')]

def deleteSpeakers(idsSpeakers):
	global speakers, speakers_key

	for idSpeaker in idsSpeakers:
		del speakers_key[idSpeaker]

	torch.save(speakers_key, args.fileSpeakers)

	speakers = list(speakers_key.keys())

	return [gr.update( value=None, choices=speakers), gr.update( value=None, choices=speakers)]



def main():
	global speakers, speakers_key

	with gr.Blocks() as demo:
		gr.Markdown('# CosyVoice 2')
		gr.Markdown('官方代码库 [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) \
					官方预训练模型 [CosyVoice2-0.5B](https://www.modelscope.cn/models/iic/CosyVoice2-0.5B) \
			  		UI重构代码库 [by DanoR](https://github.com/zheung/cosy-voice)')

		with gr.Tab('生成音频'):
			with gr.Row(equal_height=True):
				with gr.Column(scale=9):
					radioIDSpeaker = gr.Radio(label='现有音色', choices=speakers, value=speakers[0] if len(speakers) > 0 else None)
					radioModeInference = gr.Radio(label='推理功能', choices=['普通', '指令', '跨语种', '音色覆盖'], value='普通')
					with gr.Row(equal_height=True):
						audioSourceCoverTimbre = gr.Audio(label='源音频（被覆盖音色的音频）', sources='upload', type='filepath', visible=False)
						textInstruct = gr.Textbox(label='指令文本', scale=1, placeholder='如：用粤语讲这句话', value='')
						textCreate = gr.Textbox(label='合成文本', scale=3, value='我是通义实验室语音团队全新推出的生成式语音大模型，提供舒适自然的语音合成能力。')
				with gr.Column(scale=1):
					radioStream = gr.Radio(label='流式推理', choices=[('否', False), ('是', True)], value=False)
					numberSpeed = gr.Number(label='语速调节（流式推理时无效）', minimum=0.5, maximum=2.0, step=0.1, value=1)
					numberSeed = gr.Number(label='推理种子', value=0)
					buttonSeedUpdate = gr.Button(value='随机')
			buttonAudioCreate = gr.Button('生成音频', variant='primary')
			audioCreated = gr.Audio(label='生成结果', autoplay=True, streaming=True)

		with gr.Tab('复刻音色'):
			with gr.Row(equal_height=True):
				with gr.Column(scale=3, min_width=0):
					audioPromptUpload = gr.Audio(label='上传提示音频（推荐时长10~15秒，采样率应不低于16000hz）', sources='upload', type='filepath')
				with gr.Column(scale=1, min_width=0):
					audioPromptRecord = gr.Audio(label='录制提示音频', sources='microphone', type='filepath')
			textPrompt = gr.Textbox(label='提示音频文本（与提示音频内容一致）', max_lines=1, value='')
			textIDSpeakerInsert = gr.Textbox(label='创建音色ID', placeholder='不重复', max_lines=1, value='')
			buttonSpeakerInsert = gr.Button('复刻音色', variant='primary')

		with gr.Tab('管理音色'):
			boxCheckSpeakers = gr.CheckboxGroup(label='现有音色', choices=speakers)
			buttonDeleteSpeakers = gr.Button('删除', variant='stop')
			with gr.Row(equal_height=True):
				with gr.Column(scale=4, min_width=0):
					textIDSpeakerUpdate = gr.Textbox(label='改名音色ID', max_lines=1)
				with gr.Column(scale=1, min_width=0):
					buttonSpeakerUpdate = gr.Button('改名', variant='secondary')


		buttonAudioCreate.click(createAudio,
			inputs=[
				radioModeInference,
				textCreate, textInstruct, audioSourceCoverTimbre,
				radioIDSpeaker,
				textPrompt, audioPromptUpload, audioPromptRecord,
				numberSpeed, radioStream, numberSeed,
			],
			outputs=[audioCreated],
		)


		buttonSpeakerInsert.click(insertSpeaker,
			inputs=[textIDSpeakerInsert, textPrompt, audioPromptUpload, audioPromptRecord],
			outputs=[radioIDSpeaker, boxCheckSpeakers, buttonSpeakerInsert])
		buttonSpeakerUpdate.click(updateSpeaker, inputs=[boxCheckSpeakers, textIDSpeakerUpdate], outputs=[radioIDSpeaker, boxCheckSpeakers, textIDSpeakerUpdate])
		buttonDeleteSpeakers.click(deleteSpeakers, inputs=[boxCheckSpeakers], outputs=[radioIDSpeaker, boxCheckSpeakers])


		def updateTimbreCoverSourceAudioVisible(modeInference):
			return [gr.update(visible=modeInference == '音色覆盖'), gr.update(visible=modeInference != '音色覆盖'), gr.update(visible=modeInference == '指令')]
		radioModeInference.change(updateTimbreCoverSourceAudioVisible, inputs=[radioModeInference], outputs=[audioSourceCoverTimbre, textCreate, textInstruct])


		def selectSpeakers():
			global speakers, speakers_key
			speakers = list(speakers_key.keys())
			return [gr.update(choices=speakers, value=None), gr.update(choices=speakers, value=None)]
		demo.load(selectSpeakers, outputs=[radioIDSpeaker, boxCheckSpeakers])


		def updateSeed():
			return gr.update(value=random.randint(1, 100000000))
		buttonSeedUpdate.click(updateSeed, inputs=[], outputs=numberSeed)


	demo.queue(max_size=4, default_concurrency_limit=2)
	demo.launch(server_name='0.0.0.0', server_port=args.port)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--port', type=int, default=8000)
	parser.add_argument('--dirModel', type=str, default='pretrained_models/CosyVoice2-0.5B', help='local path or modelscope repo id')
	parser.add_argument('--fileSpeakers', type=str, default='speakers.pt')
	args = parser.parse_args()


	try:
		cosyvoice = CosyVoice2(args.dirModel)
		datasEmpty = np.zeros(cosyvoice.sample_rate)
	except Exception:
		raise TypeError('没有找到模型')


	if os.path.exists(args.fileSpeakers):
		speakers_key = torch.load(args.fileSpeakers, map_location=cosyvoice.frontend.device)

	speakers = list(speakers_key.keys())


	main()
