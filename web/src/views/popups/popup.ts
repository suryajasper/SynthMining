import m from 'mithril';
import '../../css/popup.scss';
import { icons } from '../icons';

const popupStatus = {
  active: false,
}

function updatePopupOverlayStatus(status: {active: boolean}) : void {
  popupStatus.active = status.active;
}

window.onkeydown = e => {
  if (e.key === 'Escape') {
    updatePopupOverlayStatus({active: false});
    m.redraw();
  }
}

interface PopupOverlayAttrs { disabledCallback(): void; }
interface PopupOverlayState { prevActive: boolean; }

const PopupOverlay : m.Component<PopupOverlayAttrs, PopupOverlayState> = {
  view(vnode) {
    if (vnode.state.prevActive !== popupStatus.active && !popupStatus.active)
      vnode.attrs.disabledCallback();
    vnode.state.prevActive = popupStatus.active;

    return m('div.popup-background-blur', {
      class: popupStatus.active ? 'enabled' : '',
    },
      vnode.children, 
    );
  },
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

abstract class Popup implements m.ClassComponent<PopupAttrs> {
  protected input: object;
  protected active: boolean;
  protected title: string;
  protected actions: PopupAction[];

  constructor() {
    this.input = {};
    this.active = false;
    this.title = '';
    this.actions = [];
  }

  createInputGroup(params: PopupInputGroupParams) {
    params.type = params.type || 'text';
    params.placeholder = params.placeholder || '';
    params.displayTitle = params.displayTitle || params.id;

    let inputSelector : string = params.type === 'textarea' ? 
      'textarea.input-group-textarea' : 'input.input-group-input';
    
    let inputAttrs = {
      name: params.id, 
      placeholder: params.placeholder,
      value: this.input[params.id] !== undefined ? this.input[params.id] : params.initialValue,
      oninput: e => {
        this.input[params.id] = e.target.value;
      },
    };

    if (params.type !== 'textarea') inputAttrs['type'] = params.type;

    return m('div.input-group', {
      class: params.type === 'textarea' ? 'long-in' : '',
    }, [
      m('label.input-group-label', { for: params.id }, params.displayTitle),

      m(inputSelector, inputAttrs),
    ]);
  }

  createCallbackButton(action: PopupAction) : m.Children {
    return m('button.popup-button', {
      onclick: () => {
        action.res();
      }
    },
      action.name, 
    )
  }

  hidePopup(vnode: {attrs: PopupAttrs}) {
    this.input = {};
    this.active = false;
    vnode.attrs.disabledCallback();
  }

  abstract loadPopupContent(vnode: {attrs: PopupAttrs}) : m.Children;

  view(vnode: {attrs: PopupAttrs}) {
    return m('div.popup-container', {
      style: { display: vnode.attrs.active ? 'flex' : 'none', },
      tabindex: 0,
      // onblur: e => this.hidePopup(vnode),
    }, [
      m('div.popup-header', [
        m('span.popup-title', this.title),

        m('button.exit-view-button', {
          title: 'Cancel',
          onclick: e => { this.hidePopup(vnode); }
        },
          icons.exit,
        )
      ]),
      m('div.popup-content', 
        this.loadPopupContent(vnode),
      ),
      m('div.popup-footer', 
        this.actions
          .map(this.createCallbackButton)
      ),
    ]);
  }
}

export { updatePopupOverlayStatus, PopupOverlay, PopupAttrs, Popup };