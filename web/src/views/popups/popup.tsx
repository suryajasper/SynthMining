import React, { useState } from 'react';
import '../../css/popup.scss';
import { icons } from '../icons';
import { PopupManagerState, usePopupStore, withPopupStore } from '../hooks/popup-state';

// const popupStatus = {
//   active: false,
// }

// function updatePopupOverlayStatus(status: {active: boolean}) : void {
//   popupStatus.active = status.active;
// }

interface PopupOverlayAttrs {
  children: JSX.Element[] | JSX.Element;
}

export function PopupOverlay(props: PopupOverlayAttrs) {
  // const [prevActive, setPrevActive] = useState(false);

  // if (prevActive !== popupStatus.active && !popupStatus.active)
  //   props.disabledCallback();
  // setPrevActive(popupStatus.active);

  const popupOverlayActive = usePopupStore(state => state.overlayActive);

  return (
    <div className={
      [
        'popup-background-blur',
        popupOverlayActive ? 'enabled' : '',
      ]
        .join(' ')
    }>
      {props.children}
    </div>
  );
}

export interface PopupAttrs {
  data: any;
  
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
  extends React.Component<PopupAttrs & { store: PopupManagerState }, PopupState> {
  
  constructor(props) {
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
    let inputTag = inputSelector.split('.')[0];
    let inputClass = inputSelector.split('.')[1];
    
    if (this.state.active && this.state.input[params.id] === undefined) {
      this.setInputValue(params.id, params.initialValue);
    }

    let inputAttrs = {
      name: params.id, 
      placeholder: params.placeholder,
      defaultValue: this.state.input[params.id] === undefined ? params.initialValue : this.state.input[params.id],
      onInput: e => {
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
        { React.createElement(inputTag, Object.assign({className: inputClass}, inputAttrs)) }
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

  abstract loadPopupContent() : JSX.Element | JSX.Element[];

  componentDidMount(): void {
    window.onkeydown = e => {
      if (e.key === 'Escape')
        this.props.store.hidePopup();
    };
  }

  render() {
    const isActive = this.props.store.overlayActive && this.props.store.activePopupId;

    return (
      <div 
        className='popup-container'
        tabIndex={0}
        style={{ display: isActive ? 'flex' : 'none', }}
      >
        <div className='popup-header'>
          <span className='popup-title'>{this.state.title}</span>

          <button 
            className='exit-view-button'
            title='Cancel'
            onClick={this.props.store.hidePopup}
          >
            {icons.exit}
          </button>

          <div className='popup-content'>{this.loadPopupContent()}</div>
        </div>

        <div className='popup-footer'>{
          this.state.actions.map(this.createCallbackButton)
        }</div>
      </div>
    );
  }
}

export default Popup;