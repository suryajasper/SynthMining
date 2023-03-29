import React, { useState } from 'react';
import '../../css/popup.scss';
import { icons } from '../icons';

const popupStatus = {
  active: false,
}

function updatePopupOverlayStatus(status: {active: boolean}) : void {
  popupStatus.active = status.active;
}

interface PopupOverlayAttrs { 
  disabledCallback(): void; 
  children: JSX.Element[] | JSX.Element;
}
interface PopupOverlayState { prevActive: boolean; }

function PopupOverlay(props: PopupOverlayAttrs) {
  const [prevActive, setPrevActive] = useState(false);

  if (prevActive !== popupStatus.active && !popupStatus.active)
    props.disabledCallback();
  setPrevActive(popupStatus.active);

  return (
    <div className={
      [
        'popup-background-blur',
        popupStatus.active ? 'enabled' : '',
      ]
        .join(' ')
    }>
      {props.children}
    </div>
  );
}

interface PopupAttrs {
  active: boolean;
  data: any;
  disabledCallback(): void;
  reloadCallback(): void;
}

interface PopupAction {
  name: string;
  res(): void;
}

interface PopupInputGroupParams { 
  id: string, 
  displayTitle?: string, 
  type?: string, 
  placeholder?: string, 
  initialValue: string,
}

interface PopupState {
  input: object;
  active: boolean;
  title: string;
  actions: PopupAction[];
}

abstract class Popup 
  extends React.Component<PopupAttrs, PopupState> {
  
  constructor(props: PopupAttrs) {
    super(props);

    this.state = {
      input: {},
      active: false,
      title: '',
      actions: [],
    };
  }

  setInputValue(inputId, value) {
    this.setState(({ input }) => {
      input[inputId] = value;
      return { input };
    })
  }

  createInputGroup(params: PopupInputGroupParams) {
    params.type = params.type || 'text';
    params.placeholder = params.placeholder || '';
    params.displayTitle = params.displayTitle || params.id;

    let inputSelector : string = params.type === 'textarea' ? 
      'textarea.input-group-textarea' : 'input.input-group-input';
    
    if (this.state.active && this.state.input[params.id] === undefined) {
      this.setInputValue(params.id, params.initialValue);
    }

    let inputAttrs = {
      name: params.id, 
      placeholder: params.placeholder,
      value: this.state.input[params.id] === undefined ? params.initialValue : this.state.input[params.id],
      oninput: e => {
        this.setInputValue(params.id, e.target.value);
      },
    };

    if (params.type !== 'textarea') inputAttrs['type'] = params.type;

    return (
      <div className={[ 'input-group',
          params.type === 'textarea' ? 'long-in' : '',
        ].join(' ')
      }>
        <label className='input-group-label' htmlFor={params.id}> {params.displayTitle} </label>
        { React.createElement(inputSelector, inputAttrs) }
      </div>
    );
  }

  createCallbackButton(action: PopupAction) : JSX.Element {
    return (
      <button 
        className='popup-button'
        onClick={action.res}
      >
        {action.name}
      </button>
    );
  }

  hidePopup() {
    this.setState({ input: {}, active: false, });
    this.props.disabledCallback();
  }

  abstract loadPopupContent() : JSX.Element | JSX.Element[];

  render() {
    if (this.props.active && !this.state.active)
      window.onkeydown = e => {
        if (e.key === 'Escape')
          this.hidePopup();
      };
    this.setState({ active: this.props.active });

    return (
      <div 
        className='popup-container'
        tabIndex={0}
        style={{ display: this.state.active ? 'flex' : 'none', }}
      >
        <div className='popup-header'>
          <span className='popup-title'>{this.state.title}</span>

          <button 
            className='exit-view-button'
            title='Cancel'
            onClick={this.hidePopup}
          >
            {icons.exit}
          </button>

          <div className='popup-content'>{this.loadPopupContent()}</div>

          <div className='popup-footer'>{
            this.state.actions.map(this.createCallbackButton)
          }</div>
        </div>
      </div>
    );
  }
}

export { updatePopupOverlayStatus, PopupOverlay, PopupAttrs, Popup };